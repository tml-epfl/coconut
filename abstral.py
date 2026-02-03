from typing import List

import random

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


def abstral_sample(sample, n_abstral_tokens: int = 50):
    idx_to_symbol = sample["idx_to_symbol"]
    abstral_tokens = get_ordered_sample(n_abstral_tokens, len(idx_to_symbol))

    sample["question"] = abstral_text(sample["question"], idx_to_symbol, abstral_tokens)
    for i in range(len(sample["steps"])):
        sample["steps"][i] = abstral_text(sample["steps"][i], idx_to_symbol, abstral_tokens)
    sample["answer"] = abstral_text(sample["answer"], idx_to_symbol, abstral_tokens)
    sample["abstral_tokens"] = abstral_tokens
    return sample
