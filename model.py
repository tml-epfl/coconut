import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


# ==========================================
# 1. CONFIGURATION
# ==========================================
class MultiHopConfig(PretrainedConfig):
    model_type = "multihop_reasoner"

    def __init__(
        self,
        vocab_size=50257,
        n_embd=256,  # d_model
        n_layer=3,  # Theoretical minimum for 2-hop reasoning
        n_head=4,
        max_position_embeddings=2048,  # RoPE cache limit
        use_mlp=False,  # Toggle: Off by default (Attention-only theory)
        use_layer_norm=True,  # Toggle: On/Off
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_position_embeddings = max_position_embeddings
        self.use_mlp = use_mlp
        self.use_layer_norm = use_layer_norm
        super().__init__(**kwargs)


# ==========================================
# 2. BASE CLASS (Must be defined before Model)
# ==========================================
class MultiHopPreTrainedModel(PreTrainedModel):
    config_class = MultiHopConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# ==========================================
# 3. HELPER MODULES (RoPE & Attention)
# ==========================================
class MinimalRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().to(dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin().to(dtype=torch.float32), persistent=False
        )

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MinimalAttentionRoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.n_embd // config.n_head
        self.num_heads = config.n_head
        self.hidden_size = config.n_embd

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = MinimalRotaryEmbedding(
            self.head_dim, config.max_position_embeddings
        )

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn_output)


class MinimalMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 4)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ==========================================
# 4. LAYERS AND BODY
# ==========================================
class MultiHopBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_layer_norm = config.use_layer_norm
        self.use_mlp = config.use_mlp
        self.attn = MinimalAttentionRoPE(config)

        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm(config.n_embd)
            if self.use_mlp:
                self.ln2 = nn.LayerNorm(config.n_embd)

        if self.use_mlp:
            self.mlp = MinimalMLP(config)

    def forward(self, hidden_states):
        residual = hidden_states
        if self.use_layer_norm:
            hidden_states = self.ln1(hidden_states)
        attn_out = self.attn(hidden_states)
        hidden_states = residual + attn_out

        if self.use_mlp:
            residual = hidden_states
            if self.use_layer_norm:
                hidden_states = self.ln2(hidden_states)
            mlp_out = self.mlp(hidden_states)
            hidden_states = residual + mlp_out
        return hidden_states


class MultiHopModel(MultiHopPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.n_embd
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.layers = nn.ModuleList(
            [MultiHopBlock(config) for _ in range(config.n_layer)]
        )
        if config.use_layer_norm:
            self.ln_f = nn.LayerNorm(self.embed_dim)
        else:
            self.ln_f = nn.Identity()
        self.post_init()

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


# ==========================================
# 5. CAUSAL LM HEAD (Use this one!)
# ==========================================
class MultiHopForCausalLM(MultiHopPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MultiHopModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights (standard for Transformers)
        self.lm_head.weight = self.model.wte.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.model.wte

    def set_input_embeddings(self, value):
        self.model.wte = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        **kwargs
    ):
        # Forward through the body
        hidden_states = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds)

        # Calculate Logits
        logits = self.lm_head(hidden_states)

        # Calculate Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states if kwargs.get("output_hidden_states") else None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Required for model.generate()
        return {"input_ids": input_ids}
