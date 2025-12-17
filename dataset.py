# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import itertools
import random
from dataclasses import dataclass
from re import I
from typing import Optional, Tuple
from tqdm import tqdm

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from preprocessing.prosqa import (
    DAG,
    sample_names_for_dag,
    generate_query_from_dag,
)


def get_dataset(path, tokenizer, max_size=1000000000):
    def tokenize_sample(sample):
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

    # verify
    d = data[0]
    complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    assert (
        complete_tokenized
        == dataset[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
        + dataset[0]["answer_tokenized"]
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

        # Check if we should include remaining (non-abstracted) steps during eval
        eval_with_visible_steps = getattr(configs, "eval_with_visible_steps", False)

        if eval_with_visible_steps:
            # Number of steps being abstracted is determined by scheduled_stage
            n_abstracted_steps = min(scheduled_stage, len(sample["steps_tokenized"]))
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
            n_latent_tokens = min(max_latent_stage, scheduled_stage) * configs.c_thought
            tokens = (
                sample["question_tokenized"]
                + ([] if no_special_marker else [start_id])
                + [latent_id] * n_latent_tokens
                + ([] if no_special_marker else [end_id])
            )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
            "graph": {
                "idx_to_symbol": sample["idx_to_symbol"],
                "edges": sample["edges"],
                "root": sample["root"],
                "target": sample["target"],
                "neg_target": sample["neg_target"],
            },
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
            "graph": {
                "idx_to_symbol": sample["idx_to_symbol"],
                "edges": sample["edges"],
                "root": sample["root"],
                "target": sample["target"],
                "neg_target": sample["neg_target"],
            },
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
    num_nodes: Tuple[int, int],
    num_layers: Tuple[int, int],
    num_edges: Tuple[int, int],
    names: str,
    entities: str,
    dist: str = "gauss",  # "gauss", "unif"
    length: int = -1,
    min_length: int = 1,
    neg_length: int = -1,
    min_neg_length: int = 1,
    num_chains: int = 1,
    max_trials: int = 100,
):
    print(f"Generating dataset with size {size} and outputting to path {path}!")
    with open(names, "r") as file:
        names = file.readlines()
    with open(entities, "r") as file:
        entities = file.readlines()

    def generate_samples(idx: int) -> dict:
        """
        Args:
            idx (int): The index of the sample to generate.

        Returns:
            dict: A dictionary with "question", "steps", and "answer" keys.
        """
        assert (
            method == "tml" or method == "prosqa"
        ), f"`method` needs to be either `tml` or `prosqa`"

        if length > 0:
            assert (
                length >= min_length
            ), f"`length` needs to be equal or larger than `min_length`"
        if neg_length > 0:
            assert (
                neg_length >= min_neg_length
            ), f"`neg_length` needs to be equal or larger than `min_neg_length`"

        if dist == "gauss":
            dist_fn = random.gauss
        elif dist == "unif":
            dist_fn = random.randint

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
                    # node_labels tracks family membership (1=family of node 0, 2=family of node 1, etc.)
                    # Only used for prosqa method
                    family_labels = None

                    if method == "tml":
                        dag = DAG.generate_layered_dag(
                            num_nodes=n_nodes,
                            num_layers=n_layers,
                            num_edges=n_edges,
                        )
                        assert (
                            sum([len(e) for e in dag.edges]) == n_edges
                        ), f"Number of edges {sum([len(e) for e in dag.edges])} is not equal to the given quantity {n_edges}!"
                        assert (
                            max(dag.layers) + 1 == n_layers
                        ), f"Number of layers {max(dag.layers) + 1} is not equal to the given quantity {n_layers}"
                    elif method == "prosqa":
                        dag, family_labels = DAG.generate_prosqa_dag(
                            num_nodes=n_nodes,
                        )
                        assert (
                            max(dag.layers) >= min_length
                        ), f"Number of layers {max(dag.layers) + 1} is not equal to or greater than the given quantity {min_length + 1}"

                    # if provided 0, sample a random length
                    max_length = max(dag.layers)
                    _length = (
                        random.randint(min_length, max_length)
                        if length == 0
                        else length
                    )
                    _neg_length = (
                        random.randint(min_neg_length, max_length)
                        if neg_length == 0
                        else neg_length
                    )

                    # entity_names are the string names for each node
                    entity_names = sample_names_for_dag(dag, names, entities)
                    nodes, context, question, chains, answer = generate_query_from_dag(
                        dag,
                        entity_names,
                        node_labels=family_labels,
                        length=_length,
                        neg_length=_neg_length,
                        num_chains=num_chains,
                    )

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
                            "graph_idx": idx,
                        }
                        for chain in chains
                    ]
                except KeyboardInterrupt:
                    raise
                except:
                    continue

    dataset = []
    sample_id_counter = itertools.count()

    for graph_idx in tqdm(range(size), desc="Generating samples"):
        batch = generate_samples(graph_idx)

        for sample in batch:
            sample["idx"] = next(sample_id_counter)

        dataset.extend(batch)

    with open(path, "w") as f:
        json.dump(dataset, f)
