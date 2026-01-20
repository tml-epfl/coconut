# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Train a custom GPT-2 style ByteLevel BPE tokenizer for synthetic TML data.

This script creates a ByteLevel BPE tokenizer (exactly like GPT-2's tokenizer)
specifically designed for the synthetic reasoning data used in coconut training.
The ByteLevel approach handles all bytes uniformly, avoiding spacing issues that
can occur with Metaspace-based tokenizers.

Usage:
    # Generate data and train tokenizer:
    python train_tokenizer.py --output_dir tokenizers/tml_custom

    # Train from existing data files:
    python train_tokenizer.py --output_dir tokenizers/tml_custom \
        --train_files data/train.json data/valid.json

    # Customize vocabulary size:
    python train_tokenizer.py --output_dir tokenizers/tml_custom --vocab_size 150
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    normalizers,
    decoders,
    AddedToken,
)
from transformers import PreTrainedTokenizerFast


def collect_texts_from_json(json_paths: List[str]) -> List[str]:
    """Extract all text from JSON data files."""
    texts = []
    for path in json_paths:
        with open(path, "r") as f:
            data = json.load(f)
        for sample in data:
            # Collect question
            texts.append(sample["question"])
            # Collect steps
            for step in sample["steps"]:
                texts.append(step)
            # Collect answer
            texts.append(sample["answer"])
    return texts


def generate_sample_texts(
    names_file: str = "data/names.txt",
    entities_file: str = "data/entities.txt",
    num_samples: int = 1000,
    just_names_entities: bool = False,
    merge_whitespace: bool = False
) -> List[str]:
    """
    Generate sample texts to train the tokenizer on.
    This covers all the vocabulary that might appear in synthetic data.
    """
    import random
    from preprocessing.prosqa import DAG, sample_names_for_dag, generate_query_from_dag

    with open(names_file, "r") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    with open(entities_file, "r") as f:
        entities = [line.strip() for line in f.readlines() if line.strip()]

    texts = []

    # Add all base vocabulary items
    texts.extend(names)
    texts.extend(entities)

    # Add common format strings used in the data
    if merge_whitespace:
        format_strings = [
            " Is ",
            " a ",
            " or a ",
            " Every ",
            " is a ",
            "###",
            "?",
            ".",
            "\n",
            " "
        ]
    else:
        format_strings = [
            "Is",
            "a",
            "is",
            "Every",
            "or",
            "###",
            "?",
            ".",
            "\n",
            " ",  # Explicitly add the space character as a token
        ]
    texts.extend(format_strings)

    if just_names_entities:
        return texts

    # Generate sample queries to capture all patterns
    print(f"Generating {num_samples} sample queries for tokenizer training...")
    for i in range(num_samples):
        try:
            # Vary the DAG parameters to cover different scenarios
            n_nodes = random.randint(5, 30)
            n_layers = random.randint(2, 8)
            n_edges = random.randint(n_layers, min(n_nodes * 2, 50))

            dag = DAG.generate_layered_dag(
                num_nodes=n_nodes,
                num_layers=n_layers,
                num_edges=n_edges,
            )

            entity_names = sample_names_for_dag(dag, names, entities)
            result = generate_query_from_dag(
                dag,
                entity_names,
                length=-1,
                neg_length=-1,
                num_chains=1,
            )

            if result is not None:
                nodes, context, question, chains, answer = result
                texts.append(context)
                texts.append(question)
                for chain in chains:
                    texts.extend(chain)
                texts.append(answer)
        except Exception:
            continue

    return texts


def train_tokenizer(
    texts: List[str],
    vocab_size: int = 200,
    min_frequency: int = 1,
) -> Tokenizer:
    """
    Train a GPT-2 style ByteLevel BPE tokenizer on the provided texts.

    This uses the exact same tokenization approach as GPT-2:
    - ByteLevel pre-tokenizer that encodes all bytes
    - ByteLevel decoder for proper detokenization
    - No unknown tokens (ByteLevel can represent any byte sequence)

    Args:
        texts: List of text strings to train on
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a token to be included

    Returns:
        Trained Tokenizer object
    """
    # Initialize BPE tokenizer (GPT-2 style with ByteLevel doesn't need unk_token
    # since it can represent any byte, but we include it for safety)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Set up normalizer (minimal normalization to preserve original text)
    # GPT-2 uses minimal normalization - just handle line endings consistently
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace("\r\n", "\n"),
            normalizers.Replace("\r", "\n"),
        ]
    )

    # Pre-tokenizer: GPT-2 style ByteLevel
    # - add_prefix_space=False: don't add space at beginning (GPT-2 default)
    # - use_regex=True: use GPT-2's regex pattern to split on whitespace boundaries
    # This splits text like: "Hello world" -> ["Hello", " world"]
    # The space is kept as part of the token, avoiding spacing issues
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder for proper detokenization (ByteLevel decoder matches the pre-tokenizer)
    tokenizer.decoder = decoders.ByteLevel()

    # Special tokens that we need for the model
    special_tokens = [
        "<pad>",  # Padding token
        "<unk>",  # Unknown token
        "<eos>",  # End of sequence
        "<|start-latent|>",  # Coconut latent start
        "<|end-latent|>",  # Coconut latent end
        "<|latent|>",  # Coconut latent token
    ]

    # Train the tokenizer with GPT-2 style settings
    # initial_alphabet is set to ByteLevel's alphabet to ensure all bytes are covered
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # Write texts to temporary file for training
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for text in texts:
            f.write(text + "\n")
        temp_path = f.name

    try:
        tokenizer.train([temp_path], trainer)
    finally:
        os.unlink(temp_path)

    return tokenizer


def train_tokenizer_2(texts: List[str], n_abstral_tokens: int = 0, merge_whitespace: bool = False) -> Tokenizer:
    # 1. Initialize empty BPE
    tokenizer = Tokenizer(models.BPE())

    # 2. IMPROVED Pre-tokenization
    # Use a regex that keeps spaces, punctuation, and words as separate pieces.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            # This regex splits on spaces and punctuation while keeping them all
            pre_tokenizers.Split(pattern=r"(\s+|[.,!?])", behavior="isolated"),
        ]
    )

    # 3. Special Tokens
    special_tokens = [
        "<pad>",
        "<unk>",
        "<eos>",
        "<|start-latent|>",
        "<|end-latent|>",
        "<|latent|>",
    ] + [f"<|node_{i}|>" for i in range(n_abstral_tokens)]
    tokenizer.add_special_tokens(special_tokens)

    # 4. Atomic Tokens
    unique_atoms = set()
    for w in texts:
        # If it's a multi-character word, add it.
        # If it's a single char (like '.'), add it.
        if merge_whitespace:
            unique_atoms.add(w)
        elif w.strip():
            unique_atoms.add(w.strip())

    unique_atoms.update([" ", ".", "?", "!", "\n"])

    atomic_tokens = []
    for w in unique_atoms:
        # RULE: Only use single_word=True for alphanumeric words (like 'Max' or 'storpus')
        # For punctuation and spaces, single_word MUST be False.
        is_alphanumeric = w.isalnum()

        atomic_tokens.append(
            AddedToken(w, single_word=is_alphanumeric, lstrip=False, rstrip=False)
        )

    tokenizer.add_tokens(atomic_tokens)

    # 5. THE DECODER (Crucial for passing the roundtrip test)
    # This tells the tokenizer how to put the pieces back together.
    tokenizer.decoder = decoders.Sequence(
        [
            decoders.Replace(
                " ", " "
            ),  # Ensures space tokens are treated as literal spaces
            # If you find you have double spaces or specific issues,
            # you can add more decoders here.
        ]
    )

    return tokenizer


def create_hf_tokenizer(
    tokenizer: Tokenizer, n_abstral_tokens: int = 0
) -> PreTrainedTokenizerFast:
    """
    Wrap the trained tokenizer in HuggingFace's PreTrainedTokenizerFast.

    Args:
        tokenizer: Trained Tokenizer object

    Returns:
        PreTrainedTokenizerFast compatible with transformers library
    """
    # Create the HuggingFace tokenizer wrapper
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token=None,  # GPT-2 style: no BOS token
        additional_special_tokens=["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
        + [f"<|node_{i}|>" for i in range(n_abstral_tokens)],
    )

    # Set padding side to right (same as GPT-2)
    hf_tokenizer.padding_side = "right"

    return hf_tokenizer


def verify_tokenizer(tokenizer: PreTrainedTokenizerFast, test_texts: List[str]):
    """Verify the tokenizer works correctly on sample texts."""
    print("\n" + "=" * 60)
    print("TOKENIZER VERIFICATION (GPT-2 Style ByteLevel BPE)")
    print("=" * 60)

    print(f"\nVocabulary size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    print("\n--- Sample tokenizations ---")
    all_passed = True
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        token_strs = tokenizer.convert_ids_to_tokens(tokens)

        # Check roundtrip
        roundtrip_ok = decoded == text
        status = "✓" if roundtrip_ok else "✗"
        if not roundtrip_ok:
            all_passed = False

        print(f"\n{status} Original: {repr(text)}")
        print(f"  Tokens:   {token_strs}")
        print(f"  IDs:      {tokens}")
        print(f"  Decoded:  {repr(decoded)}")
        if not roundtrip_ok:
            print(f"  WARNING: Roundtrip mismatch!")

    # Test spacing preservation (important for ByteLevel)
    print("\n--- Spacing preservation tests ---")
    spacing_tests = [
        "Hello world",  # Single space
        "Hello  world",  # Double space
        " Leading space",  # Leading space
        "Trailing space ",  # Trailing space
        "Line1\nLine2",  # Newline
    ]
    for text in spacing_tests:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        ok = decoded == text
        status = "✓" if ok else "✗"
        if not ok:
            all_passed = False
        print(f"{status} {repr(text)} -> {repr(decoded)}")

    if all_passed:
        print("\n✓ All roundtrip tests passed!")
    else:
        print("\n✗ Some roundtrip tests failed - check the tokenizer configuration")


def main():
    parser = argparse.ArgumentParser(
        description="Train a custom tokenizer for synthetic TML data"
    )
    parser.add_argument(
        "--method", type=str, default="bpe", help="Method to use: `tml` or `bpe`"
    )
    parser.add_argument(
        "--merge_whitespace", type=str, action="store_true", help="Merges whitespace to formatting tokens"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokenizers/tml_custom",
        help="Directory to save the trained tokenizer",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=200,
        help="Target vocabulary size (default: 200)",
    )
    parser.add_argument(
        "--train_files",
        type=str,
        nargs="*",
        default=None,
        help="JSON data files to train on (if not provided, generates sample data)",
    )
    parser.add_argument(
        "--names_file",
        type=str,
        default="data/names.txt",
        help="Path to names.txt file",
    )
    parser.add_argument(
        "--entities_file",
        type=str,
        default="data/entities.txt",
        help="Path to entities.txt file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of samples to generate for training (if not using train_files)",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=1,
        help="Minimum frequency for tokens",
    )
    parser.add_argument(
        "--n_abstral_tokens",
        type=int,
        default=0,
        help="The number of tokens for Abstral",
    )

    args = parser.parse_args()

    assert args.method in ["tml", "bpe"], "method should be either `tml` or `bpe`"

    # Collect training texts
    if args.train_files:
        print(f"Loading texts from {len(args.train_files)} files...")
        texts = collect_texts_from_json(args.train_files)
    else:
        print("Generating sample texts for tokenizer training...")

        if args.method == "bpe":
            texts = generate_sample_texts(
                names_file=args.names_file,
                entities_file=args.entities_file,
                num_samples=args.num_samples,
            )
        elif args.method == "tml":
            texts = generate_sample_texts(
                names_file=args.names_file,
                entities_file=args.entities_file,
                num_samples=args.num_samples,
                just_names_entities=True,
                merge_whitespace=args.merge_whitespace,
            )

    print(f"Collected {len(texts)} text samples")

    # Train the tokenizer
    print(f"\nTraining tokenizer with vocab_size={args.vocab_size}...")

    if args.method == "bpe":
        tokenizer = train_tokenizer(
            texts=texts,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
    elif args.method == "tml":
        tokenizer = train_tokenizer_2(
            texts=texts,
            n_abstral_tokens=args.n_abstral_tokens,
            merge_whitespace=args.merge_whitespace,
        )

    # Wrap in HuggingFace format
    hf_tokenizer = create_hf_tokenizer(
        tokenizer,
        n_abstral_tokens=args.n_abstral_tokens,
    )

    # Verify the tokenizer
    test_texts = [
        "Max is a storpus.",
        "Every storpus is a bompus.",
        "Is Max a bompus or a rompus?",
        "Max is a bompus.",
        "### bompus",
        "<|start-latent|><|latent|><|latent|><|end-latent|>",
        "Every <|node_1|> is a <|node_2|>",
    ]
    verify_tokenizer(hf_tokenizer, test_texts)

    # Save the tokenizer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(output_dir)

    print(f"\n{'=' * 60}")
    print(f"SUCCESS! GPT-2 style ByteLevel BPE tokenizer saved to: {output_dir}")
    print(f"Vocabulary size: {len(hf_tokenizer)}")
    print(f"Tokenizer type: ByteLevel BPE (like GPT-2)")
    print(f"{'=' * 60}")
    print(f"\nTo use this tokenizer in training, add to your config:")
    print(f"  tokenizer_path: {output_dir}")
    print(f"\nOr run training with:")
    print(f"  python run.py model.tokenizer_path={output_dir}")
    print(f"{'=' * 60}")

    vocab = hf_tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    print("\n--- Full Vocabulary List ---")
    for token, token_id in sorted_vocab:
        # We use repr() to see hidden characters like Ġ (space) or \n
        print(f"ID {token_id:3}: {repr(token)}")


if __name__ == "__main__":
    main()
