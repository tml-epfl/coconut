from typing import Any, List

import random
import numpy as np


def get_ordered_sample(n, x):
    """
    n: The total number of items (e.g., 50 for range [0, 49])
    x: The number of items to sample
    """
    if x > n:
        raise ValueError(f"Cannot sample {x} items from a population of {n}")

    # 1. Sample x unique numbers from range [0, n-1]
    sample = random.sample(range(n), x)

    # 2. Sort the result
    return sorted(sample)


def abstral_text(text, idx_to_symbol, abstral_tokens: List[int]):
    for abstral_token, symbol in zip(abstral_tokens, idx_to_symbol):
        text = text.replace(symbol, f"<|node_{abstral_token}|>")
    return text


def deabstral_text(text, idx_to_symbol, abstral_tokens: List[int]):
    for abstral_token, symbol in zip(abstral_tokens, idx_to_symbol):
        text = text.replace(f"<|node_{abstral_token}|>", symbol)
    return text


def abstral_sample(
    sample,
    abstral_subsample: bool = True,
    abstral_shuffle: bool = True,
    n_abstral_tokens: int = 50,
):
    idx_to_symbol = sample["idx_to_symbol"]
    abstral_tokens = list(range(n_abstral_tokens))

    if abstral_subsample:
        abstral_tokens = get_ordered_sample(n_abstral_tokens, len(idx_to_symbol))
    else:
        abstral_tokens = list(range(len(idx_to_symbol)))

    if not abstral_shuffle:
        # identify the indices of each idx_to_symbol in the text
        first_pos_of_symbol = [
            sample["question"].find(symbol) for symbol in idx_to_symbol
        ]
        argsort_indices = sorted(
            range(len(first_pos_of_symbol)), key=lambda k: first_pos_of_symbol[k]
        )
        rank_of_symbols = [0] * len(argsort_indices)
        for rank, index in enumerate(argsort_indices):
            rank_of_symbols[index] = rank
        abstral_tokens = [abstral_tokens[rank] for rank in rank_of_symbols]

    sample["question"] = abstral_text(sample["question"], idx_to_symbol, abstral_tokens)
    for i in range(len(sample["steps"])):
        sample["steps"][i] = abstral_text(
            sample["steps"][i], idx_to_symbol, abstral_tokens
        )
    sample["answer"] = abstral_text(sample["answer"], idx_to_symbol, abstral_tokens)
    sample["abstral_tokens"] = abstral_tokens
    return sample
