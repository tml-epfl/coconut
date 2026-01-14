# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import itertools
import random
from dataclasses import dataclass
from re import I
from typing import Optional, Tuple, Union, List
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from eval import generate_with_config
from preprocessing.prosqa import (
    DAG,
    sample_names_for_dag,
    generate_query_from_dag,
)
from abstral import abstral_sample


def get_dataset(path, tokenizer, abstral=False, max_size=1000000000):
    def tokenize_sample(sample):
        if abstral:
            sample = abstral_sample(sample)

        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
            "graph_idx": sample["graph_idx"]
            if "graph_idx" in sample
            else sample["idx"],
            "graph": {
                "idx_to_symbol": sample["idx_to_symbol"],
                "edges": sample["edges"],
                "root": sample["root"],
                "target": sample["target"],
                "neg_target": sample["neg_target"],
            },
        }

        return sample

    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [
                dataset.map(
                    tokenize_sample, remove_columns=list(dataset.features), num_proc=32
                )
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(
            tokenize_sample, remove_columns=list(dataset.features), num_proc=32
        )

    # verify - Note: This check may fail for tokenizers that handle
    # word boundaries differently after newlines (e.g., metaspace tokenizers)
    d = data[0]
    complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    expected = (
        dataset[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
        + dataset[0]["answer_tokenized"]
    )
    if complete_tokenized != expected:
        import warnings

        warnings.warn(
            "Tokenization mismatch: separately tokenized parts don't match jointly tokenized text. "
            "This is expected for tokenizers with metaspace behavior after newlines."
        )

    return dataset


@dataclass
class MyCollator:
    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):
        assert self.tokenizer.padding_side == "right"

        """
        Pad the batch like this to maximize the reuse of kv cache.
        E.g.,
        
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx


        ("x" is word token, "-" is pad token)
        """

        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:  # if there are continuous thoughts in the sequence
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        # Extract graphs before processing
        graphs = [feature.pop("graph", None) for feature in features]

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        # Add graphs back as a list (not tensorized)
        if graphs[0] is not None:
            batch["graph"] = graphs

        return batch


def get_question_latent_dataset(
    scheduled_stage,
    base_dataset_valid,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):
    def process_dataset(sample):
        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        else:
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )

        if configs.coconut:
            # Check if we should include remaining (non-abstracted) steps during eval
            eval_with_visible_steps = getattr(configs, "eval_with_visible_steps", False)

            if eval_with_visible_steps:
                # Number of steps being abstracted is determined by scheduled_stage
                n_abstracted_steps = min(
                    scheduled_stage, len(sample["steps_tokenized"])
                )
                # Number of latent tokens is based on abstracted steps, capped by max_latent_stage
                n_latent_tokens = (
                    min(max_latent_stage, n_abstracted_steps) * configs.c_thought
                )
                is_reversed = getattr(configs, "reversed", False)

                if is_reversed:
                    # In reversed mode, last n_abstracted_steps are abstracted
                    # So we include steps[:-n_abstracted_steps] before latent tokens
                    remaining_steps = list(
                        itertools.chain.from_iterable(
                            sample["steps_tokenized"][:-n_abstracted_steps]
                            if n_abstracted_steps > 0
                            else sample["steps_tokenized"]
                        )
                    )
                    tokens = (
                        sample["question_tokenized"]
                        + remaining_steps
                        + ([] if no_special_marker else [start_id])
                        + [latent_id] * n_latent_tokens
                        + ([] if no_special_marker else [end_id])
                    )
                else:
                    # In normal mode, first n_abstracted_steps are abstracted
                    # So we include steps[n_abstracted_steps:] after latent tokens
                    remaining_steps = list(
                        itertools.chain.from_iterable(
                            sample["steps_tokenized"][n_abstracted_steps:]
                        )
                    )
                    tokens = (
                        sample["question_tokenized"]
                        + ([] if no_special_marker else [start_id])
                        + [latent_id] * n_latent_tokens
                        + ([] if no_special_marker else [end_id])
                        + remaining_steps
                    )
            else:
                # Original logic: n_latent_tokens based on scheduled_stage capped by max_latent_stage
                n_latent_tokens = (
                    min(max_latent_stage, scheduled_stage) * configs.c_thought
                )
                tokens = (
                    sample["question_tokenized"]
                    + ([] if no_special_marker else [start_id])
                    + [latent_id] * n_latent_tokens
                    + ([] if no_special_marker else [end_id])
                )
        else:
            tokens = sample["question_tokenized"]

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
            "graph": sample["graph"],
        }

    return base_dataset_valid.map(
        process_dataset, remove_columns=list(base_dataset_valid.features), num_proc=32
    )


def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
):
    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000  # skip all
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )

        else:
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        if configs.no_cot:
            n_skip_steps = 100  # skip all step
            n_latent_tokens = 0

        n_latent_tokens *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + (
                list(
                    itertools.chain.from_iterable(
                        sample["steps_tokenized"][:-n_skip_steps]
                    )
                )
                if configs.reversed
                else []
            )
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + (
                list(
                    itertools.chain.from_iterable(
                        sample["steps_tokenized"][n_skip_steps:]
                    )
                )
                if (not configs.reversed)
                else []
            )
            + sample["answer_tokenized"]
        )
        labels = [-100] * len(sample["question_tokenized"]) + tokens[
            len(sample["question_tokenized"]) :
        ]
        labels = [
            -100 if token in [start_id, latent_id, end_id] else token
            for token in labels
        ]

        return {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "graph_idx": sample["graph_idx"],
            "position_ids": list(range(len(tokens))),
            "graph": sample["graph"],
        }

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset, remove_columns=list(base_dataset.features), num_proc=32
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        processed_dataset = base_dataset.map(
            process_dataset, remove_columns=list(base_dataset.features), num_proc=32
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset


def generate_dataset(
    path: str,
    size: int,
    method: str,
    num_nodes: Union[List[Tuple[int, int]], Tuple[int, int]],
    num_layers: Union[List[Tuple[int, int]], Tuple[int, int]],
    num_edges: Union[List[Tuple[int, int]], Tuple[int, int]],
    names: str,
    entities: str,
    dist: str = "gauss",  # or "unif"
    length: int = -1,
    min_length: int = 1,
    max_length: int = -1,
    neg_length: int = -1,
    min_neg_length: int = 1,
    max_neg_length: int = -1,
    epochs_per_length: int = 0,
    num_chains: int = 1,
    max_trials: int = 100,
    teacher: torch.nn.Module = None,
    distillation_config: dict = None,
    epoch: int = 0,
):
    if not isinstance(num_nodes[0], int):
        assert (
            epochs_per_length > 0
        ), "`epochs_per_length` should be larger than 0 for multi-stage training"

        index = min(len(num_nodes) - 1, epoch // epochs_per_length)
        num_nodes = num_nodes[index]
        num_layers = num_layers[index]
        num_edges = num_edges[index]

    if epochs_per_length > 0:
        if max_length > 0:
            max_length += epoch // epochs_per_length
        if max_neg_length > 0:
            max_neg_length += epoch // epochs_per_length

    print(f"Generating dataset with size {size} and outputting to path {path}!")
    print(
        f"Parameters: num_nodes - {num_nodes}, num_layers - {num_layers}, num_edges - {num_edges}, length - {length}, min_length - {min_length}, max_length - {max_length}, neg_length - {neg_length}, min_neg_length - {min_neg_length}, max_neg_length - {max_neg_length}"
    )
    with open(names, "r") as file:
        names_list = file.readlines()
    with open(entities, "r") as file:
        entities_list = file.readlines()

    # Prepare distillation config defaults
    if distillation_config is None:
        distillation_config = {}

    args_list = [
        (
            epoch,
            graph_idx,
            method,
            num_nodes,
            num_layers,
            num_edges,
            names_list,
            entities_list,
            dist,
            length,
            min_length,
            max_length,
            neg_length,
            min_neg_length,
            max_neg_length,
            num_chains,
            max_trials,
            teacher,
            distillation_config,
        )
        for graph_idx in range(size)
    ]

    dataset = []
    sample_id_counter = itertools.count()

    with Pool(processes=cpu_count()) as pool:
        for batch in tqdm(
            pool.imap_unordered(_generate_samples_for_graph, args_list),
            total=size,
            desc="Generating samples",
        ):
            for sample in batch:
                sample["idx"] = next(sample_id_counter)
            dataset.extend(batch)

    with open(path, "w") as f:
        json.dump(dataset, f)

    return dataset


def _generate_samples_for_graph(args):
    (
        epoch,
        graph_idx,
        method,
        num_nodes,
        num_layers,
        num_edges,
        names,
        entities,
        dist,
        length,
        min_length,
        max_length,
        neg_length,
        min_neg_length,
        max_neg_length,
        num_chains,
        max_trials,
        teacher,
        distillation_config,
    ) = args

    # --- this is essentially your current generate_samples body ---
    assert method in ("tml", "prosqa"), "`method` needs to be either `tml` or `prosqa`"

    if length > 0:
        assert length >= min_length
    if neg_length > 0:
        assert neg_length >= min_neg_length
    if max_length > 0:
        assert length <= max_length
    if max_neg_length > 0:
        assert neg_length <= max_neg_length

    if dist == "gauss":
        dist_fn = random.gauss
    elif dist == "unif":
        dist_fn = random.randint
    else:
        raise ValueError(f"Unknown dist {dist}")

    while True:
        n_nodes = round(dist_fn(num_nodes[0], num_nodes[1]))
        n_layers = round(dist_fn(num_layers[0], num_layers[1]))
        n_edges = round(dist_fn(num_edges[0], num_edges[1]))

        if n_layers <= min_length:
            n_layers = min_length + 1
        if n_layers <= min_neg_length:
            n_layers = min_neg_length + 1

        for _ in range(max_trials):
            try:
                family_labels = None

                if method == "tml":
                    dag = DAG.generate_layered_dag(
                        num_nodes=n_nodes,
                        num_layers=n_layers,
                        num_edges=n_edges,
                    )
                    assert (
                        sum(len(e) for e in dag.edges) == n_edges
                    ), f"Number of edges {n_edges} do not match {sum(len(e) for e in dag.edges)}"
                    assert max(dag.layers) + 1 == n_layers
                else:  # prosqa
                    dag, family_labels = DAG.generate_prosqa_dag(
                        num_nodes=n_nodes,
                    )
                    assert max(dag.layers) >= min_length

                _max_length = (
                    min(max_length, max(dag.layers))
                    if max_length > 0
                    else max(dag.layers)
                )
                _max_neg_length = (
                    min(max_neg_length, max(dag.layers))
                    if max_neg_length > 0
                    else max(dag.layers)
                )
                _length = (
                    random.randint(min_length, _max_length) if length == 0 else length
                )
                _neg_length = (
                    random.randint(min_neg_length, _max_neg_length)
                    if neg_length == 0
                    else neg_length
                )

                entity_names = sample_names_for_dag(dag, names, entities)
                nodes, context, question, chains, answer = generate_query_from_dag(
                    dag,
                    entity_names,
                    node_labels=family_labels,
                    length=_length,
                    neg_length=_neg_length,
                    num_chains=num_chains,
                )

                if teacher is not None:
                    # Generate num_chains times with teacher
                    teacher_model, teacher_tokenizer = teacher

                    # Prepare input for teacher
                    full_question = context + " " + question
                    input_ids = teacher_tokenizer.encode(
                        full_question + "\n",
                        add_special_tokens=True,
                        return_tensors="pt",
                    )

                    # Move to same device as teacher model
                    device = next(teacher_model.parameters()).device
                    input_ids = input_ids.to(device)

                    # Get generation parameters from config
                    sampling_strategy = distillation_config.get(
                        "distillation_sampling_strategy", "sample"
                    )
                    temperature = distillation_config.get(
                        "distillation_temperature", 0.7
                    )
                    top_p = distillation_config.get("distillation_top_p", 0.9)
                    num_beams = distillation_config.get("distillation_num_beams", None)
                    max_new_tokens = distillation_config.get(
                        "distillation_max_new_tokens", 128
                    )

                    # Determine how many sequences to generate
                    # If num_chains is -1 (all chains), use a default value for teacher generation
                    teacher_num_chains = num_chains if num_chains > 0 else 5

                    # Generate teacher_num_chains sequences from teacher
                    with torch.no_grad():
                        generated_sequences = generate_with_config(
                            model=teacher_model,
                            input_ids=input_ids,
                            tokenizer=teacher_tokenizer,
                            max_new_tokens=max_new_tokens,
                            num_return_sequences=teacher_num_chains,
                            sampling_strategy=sampling_strategy,
                            temperature=temperature,
                            top_p=top_p,
                            num_beams=num_beams,
                            pad_token_id=teacher_tokenizer.eos_token_id,
                            synced_gpus=False,
                        )

                    # Parse each generation and collect all teacher chains (including incorrect ones)
                    teacher_chains = []
                    # Handle shape: if num_return_sequences > 1, generated_sequences is [N, seq_len]
                    # If num_return_sequences == 1, ensure it's [1, seq_len] for consistent iteration
                    if generated_sequences.dim() == 1:
                        generated_sequences = generated_sequences.unsqueeze(0)

                    for seq in generated_sequences:
                        text_output = teacher_tokenizer.decode(
                            seq, skip_special_tokens=True
                        )

                        # Parse steps and answer from output
                        # Format: question\nstep1\nstep2\n...\n### answer
                        lines = text_output.split("\n")
                        # Skip the question (first line), get steps before "###"
                        steps_text = "\n".join(lines[1:]).split("#")[0].strip()
                        teacher_steps = [
                            s.strip() for s in steps_text.split("\n") if s.strip()
                        ]

                        # Keep all teacher-generated steps (including incorrect ones)
                        if teacher_steps:
                            teacher_chains.append(teacher_steps)

                    # Remove duplicate chains (convert to tuples for hashability, then back to lists)
                    unique_teacher_chains = [
                        list(t)
                        for t in dict.fromkeys(tuple(chain) for chain in teacher_chains)
                    ]

                    # Use unique teacher chains if we got any, otherwise fall back to original
                    if unique_teacher_chains:
                        chains = unique_teacher_chains

                return [
                    {
                        "edges": [
                            (i, item)
                            for i, sublist in enumerate(dag.edges)
                            for item in sublist
                        ],
                        "root": nodes[0],
                        "target": nodes[1],
                        "neg_target": nodes[2],
                        "idx_to_symbol": entity_names,
                        "question": context + " " + question,
                        "steps": chain,
                        "answer": answer,
                        "graph_idx": graph_idx,
                    }
                    for chain in chains
                ]
            except KeyboardInterrupt:
                raise
            except Exception as e:
                continue
