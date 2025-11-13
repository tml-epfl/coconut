# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import atexit
import datetime
import functools
import gc
import itertools
import json
import os
import sys
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hydra
import torch
import torch.distributed
import torch.distributed as dist
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from tqdm import tqdm

from coconut import Coconut
from dataset import (
    MyCollator,
    generate_dataset,
    get_cot_latent_dataset,
    get_dataset,
    get_question_latent_dataset,
)
from utils import (
    compute_experiment_signature,
    generate_attempt_id,
    get_git_metadata,
    set_seed,
)

MANIFEST_FILENAME = "run_manifest.yaml"
CONFIG_SNAPSHOT_FILENAME = "config.yaml"


def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _to_container(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _load_yaml_dict(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = OmegaConf.load(str(path))
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Warning: failed to load manifest at {path}: {exc}")
        return None
    if data is None:
        return None
    return _to_container(data)


def _save_yaml_dict(data: Dict[str, Any], path: Path) -> None:
    cfg = OmegaConf.create(data)
    OmegaConf.save(config=cfg, f=str(path), resolve=True)


def _list_checkpoints(attempt_path: Path) -> List[Tuple[int, Path]]:
    checkpoints: List[Tuple[int, Path]] = []
    if not attempt_path.exists():
        return checkpoints
    for candidate in attempt_path.glob("checkpoint_*"):
        name = candidate.name
        parts = name.split("_")
        if len(parts) != 2:
            continue
        epoch_str = parts[1]
        if not epoch_str.isdigit():
            continue
        checkpoints.append((int(epoch_str), candidate))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def _list_attempts(signature_root: Path) -> List[Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    if not signature_root.exists():
        return attempts
    for child in signature_root.iterdir():
        if not child.is_dir():
            continue
        manifest = _load_yaml_dict(child / MANIFEST_FILENAME) or {}
        checkpoints = _list_checkpoints(child)
        try:
            mtime = child.stat().st_mtime
        except OSError:
            mtime = 0
        attempts.append(
            {
                "id": child.name,
                "path": child,
                "manifest": manifest,
                "checkpoints": checkpoints,
                "mtime": mtime,
            }
        )
    attempts.sort(key=lambda item: item["mtime"], reverse=True)
    return attempts


def _prepare_attempt(
    *,
    configs: DictConfig,
    signature: str,
    resume_mode: str,
    resume_attempt_id: Optional[str],
    git_metadata: Dict[str, Any],
    signature_root: Path,
) -> Dict[str, Any]:
    resume_mode = (resume_mode or "auto").lower()
    if resume_mode not in {"auto", "force", "never"}:
        raise ValueError(
            f"Unsupported resume_mode '{resume_mode}'. "
            "Expected one of ['auto', 'force', 'never']."
        )

    signature_root.mkdir(parents=True, exist_ok=True)
    attempts = _list_attempts(signature_root)

    selected: Optional[Dict[str, Any]] = None
    if resume_attempt_id:
        for attempt in attempts:
            if attempt["id"] == resume_attempt_id:
                selected = attempt
                break
        if selected is None:
            raise ValueError(
                f"resume_attempt_id='{resume_attempt_id}' not found under {signature_root}"
            )
    elif resume_mode == "force" and attempts:
        selected = attempts[0]
    elif resume_mode == "auto":
        for attempt in attempts:
            status = attempt["manifest"].get("status", "unknown")
            if status != "completed":
                selected = attempt
                break

    if resume_mode == "never" and not resume_attempt_id:
        selected = None

    is_new_attempt = selected is None

    if is_new_attempt:
        attempt_id = generate_attempt_id()
        while (signature_root / attempt_id).exists():
            attempt_id = generate_attempt_id()
        attempt_path = signature_root / attempt_id
        attempt_path.mkdir(parents=True, exist_ok=False)
        manifest: Dict[str, Any] = {
            "signature": signature,
            "attempt_id": attempt_id,
            "status": "running",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "git": git_metadata,
            "resume_mode": resume_mode,
            "resumptions": 0,
            "last_checkpoint": None,
            "resume_epoch": 0,
        }
        checkpoints: List[Tuple[int, Path]] = []
    else:
        attempt_id = selected["id"]
        attempt_path = selected["path"]
        attempt_path.mkdir(parents=True, exist_ok=True)
        checkpoints = selected["checkpoints"]
        manifest = selected["manifest"] or {}
        manifest.setdefault("signature", signature)
        manifest["attempt_id"] = attempt_id
        manifest["status"] = "running"
        manifest["updated_at"] = _now_iso()
        manifest["git"] = git_metadata
        manifest["resume_mode"] = resume_mode
        manifest["resumptions"] = manifest.get("resumptions", 0) + 1
        resume_epoch_default = manifest.get("resume_epoch", 0)
        if checkpoints:
            resume_epoch_default = checkpoints[-1][0]
            manifest["last_checkpoint"] = checkpoints[-1][1].name
        manifest["resume_epoch"] = resume_epoch_default

    manifest_path = attempt_path / MANIFEST_FILENAME
    _save_yaml_dict(manifest, manifest_path)

    checkpoint_entries = [ckpt[1].name for ckpt in checkpoints]
    latest_checkpoint_path = str(checkpoints[-1][1]) if checkpoints else None
    resume_epoch = manifest.get("resume_epoch", 0)

    config_snapshot_path = attempt_path / CONFIG_SNAPSHOT_FILENAME
    _save_yaml_dict(_to_container(configs), config_snapshot_path)

    return {
        "attempt_id": attempt_id,
        "attempt_path": str(attempt_path),
        "manifest_path": str(manifest_path),
        "config_snapshot_path": str(config_snapshot_path),
        "is_new_attempt": is_new_attempt,
        "resume_epoch": resume_epoch,
        "latest_checkpoint_path": latest_checkpoint_path,
        "checkpoint_entries": checkpoint_entries,
    }


def _update_manifest(path: Path, updates: Dict[str, Any]) -> None:
    existing = _load_yaml_dict(path) or {}
    existing.update(updates)
    existing["updated_at"] = _now_iso()
    _save_yaml_dict(existing, path)


def _cleanup_process_group():
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            if os.environ.get("RANK", "0") == "0":
                print(f"Process group cleanup encountered an error: {exc}")


atexit.register(_cleanup_process_group)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # init distributed environment
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print("Config:", OmegaConf.to_container(cfg, resolve=True))

    merged_dict = {}
    run_section = None
    has_sections = False
    for section in ["run", "data", "method", "model", "training"]:
        if section in cfg:
            has_sections = True
            section_dict = OmegaConf.to_container(cfg[section], resolve=True)
            if section == "run":
                run_section = section_dict
            merged_dict.update(section_dict)

    if not has_sections:
        raise ValueError("No configuration sections found to merge.")

    if run_section is not None:
        merged_dict["run"] = run_section

    configs = OmegaConf.create(merged_dict)

    def has_load_path(path):
        if path is None:
            return False
        if isinstance(path, str) and path.lower() == "none":
            return False
        return str(path).strip() != ""

    set_seed(configs.run.seed)

    signature_ignore_keys = set(getattr(configs, "signature_ignore_keys", []))
    signature_ignore_keys.update(
        {
            "resume",
            "run.resume",
            "load_model_path",
            "run.load_model_path",
            "resume_mode",
            "run.resume_mode",
            "resume_attempt_id",
            "run.resume_attempt_id",
        }
    )

    git_metadata = get_git_metadata()
    experiment_signature = compute_experiment_signature(
        configs,
        ignore_keys=signature_ignore_keys,
        git_metadata=git_metadata,
        extra={"name": configs.name},
    )

    resume_mode = getattr(configs, "resume_mode", "auto")
    resume_attempt_id = getattr(configs, "resume_attempt_id", None)

    signature_root = Path(configs.save_path) / configs.name / experiment_signature

    attempt_info: Optional[Dict[str, Any]] = None
    if rank == 0:
        attempt_info = _prepare_attempt(
            configs=configs,
            signature=experiment_signature,
            resume_mode=resume_mode,
            resume_attempt_id=resume_attempt_id,
            git_metadata=git_metadata,
            signature_root=signature_root,
        )

    attempt_payload: List[Optional[Dict[str, Any]]] = [attempt_info]
    dist.broadcast_object_list(attempt_payload, src=0)
    attempt_info = attempt_payload[0]
    if attempt_info is None:
        raise RuntimeError("Failed to prepare run attempt information.")

    save_dir = attempt_info["attempt_path"]
    manifest_path = attempt_info["manifest_path"]
    config_snapshot_path = attempt_info["config_snapshot_path"]
    latest_checkpoint_path = attempt_info["latest_checkpoint_path"]
    resume_epoch = attempt_info["resume_epoch"] or 0

    if rank != 0:
        os.makedirs(save_dir, exist_ok=True)

    torch.distributed.barrier()

    manifest_path_obj = Path(manifest_path)
    manifest_completion = {"completed": False}
    if rank == 0:

        def _ensure_manifest_failure():
            if not manifest_completion["completed"]:
                _update_manifest(manifest_path_obj, {"status": "failed"})

        atexit.register(_ensure_manifest_failure)

    configs.experiment_signature = experiment_signature
    configs.attempt_id = attempt_info["attempt_id"]
    configs.attempt_path = save_dir
    configs.manifest_path = manifest_path
    configs.signature_root = str(signature_root)
    configs.config_snapshot_path = config_snapshot_path
    if "run" in configs and isinstance(configs.run, DictConfig):
        configs.run.experiment_signature = experiment_signature
        configs.run.attempt_id = attempt_info["attempt_id"]
        configs.run.attempt_path = save_dir
        configs.run.signature_root = str(signature_root)
        configs.run.config_snapshot_path = config_snapshot_path

    if configs.data_type == "synthetic":
        synthetic_data_dir = Path(save_dir) / "data"
        if rank == 0:
            synthetic_data_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        train_path = synthetic_data_dir / "train.json"
        val_path = synthetic_data_dir / "validation.json"
        configs.train_path = str(train_path)
        configs.val_path = str(val_path)
        if "run" in configs and isinstance(configs.run, DictConfig):
            configs.run.train_path = configs.train_path
            configs.run.val_path = configs.val_path

    user_requested_resume = configs.resume != 0
    if rank == 0:
        attempt_message = (
            "Starting new attempt"
            if attempt_info.get("is_new_attempt", False)
            else "Using existing attempt"
        )
        print(
            f"{attempt_message} '{attempt_info['attempt_id']}' "
            f"for signature '{experiment_signature}' at '{save_dir}'."
        )
    if latest_checkpoint_path and not configs.only_eval:
        if rank == 0:
            banner = "=" * 80
            print(
                f"\n{banner}\n"
                "!! RESUMING FROM PREVIOUS RUN !!\n"
                "Found an existing attempt with checkpoints; ignoring the provided `resume` value.\n"
                f"{banner}\n"
            )
        configs.resume = resume_epoch
        configs.load_model_path = latest_checkpoint_path
        if "run" in configs and isinstance(configs.run, DictConfig):
            configs.run.resume = resume_epoch
            configs.run.load_model_path = latest_checkpoint_path
        print(f"Loading from previous run epoch_{configs.resume}!")
    elif user_requested_resume and configs.resume != 0:
        if not has_load_path(configs.load_model_path):
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model_init = getattr(configs, "model_init", "pretrained")
    model_config_overrides = getattr(configs, "model_config_overrides", None) or {}

    if model_init not in ["pretrained", "scratch"]:
        raise ValueError(
            f"Unsupported model_init option: {model_init}. "
            "Expected one of ['pretrained', 'scratch']."
        )

    if model_init == "scratch":
        model_config = AutoConfig.from_pretrained(configs.model_id)
        for key, value in model_config_overrides.items():
            setattr(model_config, key, value)
        # keep tokenizer and model vocab sizes aligned before extra tokens
        model_config.vocab_size = len(tokenizer)
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        if model_config_overrides:
            raise ValueError(
                "model_config_overrides is only supported when model_init is 'scratch'."
            )
        model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    loaded = False

    if has_load_path(configs.load_model_path):
        saved_weights = torch.load(
            configs.load_model_path, map_location=torch.device(rank)
        )

        if configs.coconut and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # we are loading a base model into coconut model
            # e.g., for GSM8k, we used a SFTed model to skip the stage 0
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # loading from preempted run
            # will handle later
            pass

        else:
            # resume or evaluate sft model
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        # if we need new tokens, initialize their embeddings and lm heads
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        # initialize the new token embeddings with a known token
        # it helps stablize the training
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding
            # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    if has_load_path(configs.load_model_path) and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # GPT2Block,       # for GPT2, we don't need to shard layers (it becomes DDP)
            LlamaDecoderLayer  # only shard llama's layers.
        },
    )

    if configs.bf16:
        model.to(torch.bfloat16)

    # if only eval, use ddp (to avoid bugs in fsdp)
    if configs.only_eval:
        parallel_model = DDP(model, device_ids=[rank])

    else:
        parallel_model = FSDP(
            model, auto_wrap_policy=llama_auto_wrap_policy, device_id=rank
        )

    del model

    if rank == 0:
        print(parallel_model)

    if configs.data_type == "synthetic":

        def _generate_dataset(path, **configs):
            if torch.cuda.device_count() > 1:
                if dist.get_rank() == 0:
                    processed_dataset = [
                        generate_dataset(
                            path,
                            **configs,
                            names="data/names.txt",
                            entities="data/entities.txt",
                        )
                    ]
                else:
                    processed_dataset = [None]
                dist.broadcast_object_list(processed_dataset, src=0)
                dataset = processed_dataset[0]
            else:
                dataset = generate_dataset(
                    path,
                    **configs,
                    names="data/names.txt",
                    entities="data/entities.txt",
                )
            return dataset

        configs_valid = configs.dataset.copy()
        configs_valid["size"] = abs(configs_valid["size"]["valid"])
        _generate_dataset(configs.val_path, **configs_valid)

        if not configs.only_eval:
            configs_train = configs.dataset.copy()
            configs_train["size"] = abs(configs_train["size"]["train"])
            _generate_dataset(configs.train_path, **configs_train)

    # prepare the ground truth answer and cot for evaluation
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path,
            tokenizer,
            max_size=5000 if configs.debug else 100000000,
        )

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0

    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_entity = getattr(configs, "wandb_entity", None)
        if isinstance(wandb_entity, str) and wandb_entity.lower() == "none":
            wandb_entity = None
        wandb_entity = wandb_entity or os.environ.get("WANDB_ENTITY")

        wandb_kwargs = {"project": configs.project, "name": configs.name}
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity

        wandb_run = wandb.init(**wandb_kwargs)
        wandb_run.config.update(
            OmegaConf.to_container(configs, resolve=True), allow_val_change=True
        )
        wandb_run.config.update(
            {
                "experiment_signature": experiment_signature,
                "attempt_id": attempt_info["attempt_id"],
                "resume_mode": resume_mode,
                "resume_attempt_id": resume_attempt_id,
            },
            allow_val_change=True,
        )
        if config_snapshot_path and os.path.exists(config_snapshot_path):
            wandb_run.save(
                config_snapshot_path,
                base_path=os.path.dirname(config_snapshot_path),
            )
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None

    if configs.reset_optimizer:
        optimizer = None

    else:
        optimizer = optim.AdamW(
            parallel_model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    best_acc = 0

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    for epoch in range(configs.resume, configs.num_epochs):
        if (
            configs.data_type == "synthetic"
            and epoch > configs.resume
            and configs.dataset.get("online", False)
        ):
            _generate_dataset(configs.val_path, **configs_valid)
            base_dataset_valid = get_dataset(
                configs.val_path,
                tokenizer,
                max_size=32 if configs.debug else 100000000,
            )
            if not configs.only_eval:
                _generate_dataset(configs.train_path, **configs_train)
                base_dataset_train = get_dataset(
                    configs.train_path,
                    tokenizer,
                    max_size=5000 if configs.debug else 100000000,
                )

        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not configs.only_eval:
            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            # the sampler is deterministic even if shuffle is set to True
            # so we have shuffled the dataset when it's constructed (at every epoch).

            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            if configs.reset_optimizer:
                del optimizer

                optimizer = optim.AdamW(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            parallel_model.module.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):
                if step == 0 and wandb_run and rank == 0:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    # copy the table due to a bug in wandb
                    # https://github.com/wandb/wandb/issues/2981

                    wandb_run.log({"data_table": copy(text_table)})

                total_train_steps += 1
                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                }

                outputs = parallel_model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            dist.barrier()

            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                states = parallel_model.state_dict()
                if rank == 0:
                    checkpoint_name = f"checkpoint_{epoch + 1}"
                    torch.save(states, os.path.join(save_dir, checkpoint_name))
                    print("saving model.")
                    _update_manifest(
                        manifest_path_obj,
                        {"last_checkpoint": checkpoint_name, "resume_epoch": epoch + 1},
                    )

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

            # val loss
            total_loss = 0

            with torch.no_grad():
                parallel_model.module.eval()
                for step, batch in enumerate(valid_loss_dataloader):
                    batch = {
                        key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                    }

                    outputs = parallel_model(**batch)
                    loss = outputs.loss
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    total_loss += loss.item() / world_size

                if wandb_run and rank == 0:
                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

        # val generation accuracy
        total_length = len(valid_gen_dataloader)

        pbar = tqdm(
            colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
        )
        cor, cor_cot, total = (
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
        )

        with torch.no_grad():
            parallel_model.module.eval()
            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"][0]

                batch = {
                    k: v.to(rank)
                    for k, v in batch.items()
                    if v != None and k not in ["idx", "position_ids"]
                }
                # https://github.com/huggingface/transformers/issues/32492

                assert len(batch["input_ids"]) == 1
                answer = answers_val[test_idx.cpu().item()]
                answer_cot = cot_val[test_idx.cpu().item()]
                question = question_val[test_idx.cpu().item()]

                total += 1

                # synced_gpus=True in FSDP mode, as we need to keep # forward pass the same on each device
                outputs = parallel_model.module.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    synced_gpus=not configs.only_eval,
                )

                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_output = text_output.split("#")[-1].replace(",", "").strip()
                cot_output = (
                    ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                )

                if idx < 5 and rank == 0:
                    # print some examples
                    print(
                        f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'"
                    )
                    print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                    print(f"Extracted Output: '{answer_output}'")

                cor += answer_output == answer
                cor_cot += cot_output == answer_cot

                pbar.update(1)
                pbar.set_description(
                    f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                )

            pbar.close()
            print(f"Device {rank}: Cor={cor}, CoT={cor_cot}, Total={total}")

        dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

        cor_cot = cor_cot.item()
        cor = cor.item()
        total = total.item()
        if rank == 0:
            print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
            print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        sys.stdout.flush()

        if wandb_run:
            wandb_run.log({"eval/acc": cor / total, "eval/cot_em": cor_cot / total})

        if configs.only_eval:
            break

        dist.barrier()
        if (
            cor / total > best_acc
            and configs.save_only_improve
            and not configs.debug
            and not configs.only_eval
        ):
            states = parallel_model.state_dict()

            if rank == 0:
                checkpoint_name = f"checkpoint_{epoch + 1}"
                torch.save(states, os.path.join(save_dir, checkpoint_name))
                print("saving model.")
                _update_manifest(
                    manifest_path_obj,
                    {"last_checkpoint": checkpoint_name, "resume_epoch": epoch + 1},
                )

            best_acc = cor / total

            dist.barrier()
            del states
            gc.collect()
            torch.cuda.empty_cache()

    if rank == 0:
        _update_manifest(manifest_path_obj, {"status": "completed"})
        manifest_completion["completed"] = True


if __name__ == "__main__":
    main()
