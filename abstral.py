def abstral_text(text, idx_to_symbol):
    for idx, symbol in enumerate(idx_to_symbol):
        text = text.replace(symbol, f"<|node_{idx}|>")
    return text


def deabstral_text(text, idx_to_symbol):
    for idx, symbol in enumerate(idx_to_symbol):
        text = text.replace(f"<|node_{idx}|>", symbol)
    return text


def abstral_sample(sample):
    idx_to_symbol = sample["idx_to_symbol"]
    sample["question"] = abstral_text(sample["question"], idx_to_symbol)
    for i in range(len(sample["steps"])):
        sample["steps"][i] = abstral_text(sample["steps"][i], idx_to_symbol)
    sample["answer"] = abstral_text(sample["answer"], idx_to_symbol)
    return sample
