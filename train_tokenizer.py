# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Train a custom tokenizer for synthetic TML data.

This script creates a minimal vocabulary tokenizer specifically designed for
the synthetic reasoning data used in coconut training. The resulting tokenizer
has a much smaller vocabulary (typically 100-300 tokens) compared to GPT-2's
50k+ tokens, making the model significantly more efficient.

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
)
from tokenizers.processors import TemplateProcessing
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
    format_strings = [
        "is a",
        "Every",
        "or a",
        "###",
        "?",
        ".",
        "\n",
    ]
    texts.extend(format_strings)

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
    Train a BPE tokenizer on the provided texts.

    Args:
        texts: List of text strings to train on
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a token to be included

    Returns:
        Trained Tokenizer object
    """
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Set up normalizer (basic unicode normalization)
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.Replace("\r\n", "\n"),
            normalizers.Replace("\r", "\n"),
        ]
    )

    # Pre-tokenizer: split on whitespace and punctuation, keeping track of spaces
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Metaspace(replacement="▁"),
            pre_tokenizers.Punctuation(behavior="isolated"),
        ]
    )

    # Decoder for proper detokenization (matches the Metaspace pre-tokenizer)
    tokenizer.decoder = decoders.Metaspace(replacement="▁")

    # Special tokens that we need for the model
    special_tokens = [
        "<pad>",  # Padding token
        "<unk>",  # Unknown token
        "<eos>",  # End of sequence
        "<|start-latent|>",  # Coconut latent start
        "<|end-latent|>",  # Coconut latent end
        "<|latent|>",  # Coconut latent token
    ]

    # Train the tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Write texts to temporary file for training
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for text in texts:
            f.write(text + "\n")
        temp_path = f.name

    try:
        tokenizer.train([temp_path], trainer)
    finally:
        os.unlink(temp_path)

    return tokenizer


def create_hf_tokenizer(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
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
        additional_special_tokens=["<|start-latent|>", "<|end-latent|>", "<|latent|>"],
    )

    # Set padding side to right (same as GPT-2)
    hf_tokenizer.padding_side = "right"

    return hf_tokenizer


def verify_tokenizer(tokenizer: PreTrainedTokenizerFast, test_texts: List[str]):
    """Verify the tokenizer works correctly on sample texts."""
    print("\n" + "=" * 60)
    print("TOKENIZER VERIFICATION")
    print("=" * 60)

    print(f"\nVocabulary size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    print("\n--- Sample tokenizations ---")
    for text in test_texts[:5]:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        token_strs = tokenizer.convert_ids_to_tokens(tokens)
        print(f"\nOriginal: {repr(text)}")
        print(f"Tokens:   {token_strs}")
        print(f"IDs:      {tokens}")
        print(f"Decoded:  {repr(decoded)}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a custom tokenizer for synthetic TML data"
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

    args = parser.parse_args()

    # Collect training texts
    if args.train_files:
        print(f"Loading texts from {len(args.train_files)} files...")
        texts = collect_texts_from_json(args.train_files)
    else:
        print("Generating sample texts for tokenizer training...")
        texts = generate_sample_texts(
            names_file=args.names_file,
            entities_file=args.entities_file,
            num_samples=args.num_samples,
        )

    print(f"Collected {len(texts)} text samples")

    # Train the tokenizer
    print(f"\nTraining tokenizer with vocab_size={args.vocab_size}...")
    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    # Wrap in HuggingFace format
    hf_tokenizer = create_hf_tokenizer(tokenizer)

    # Verify the tokenizer
    test_texts = [
        "Max is a storpus.",
        "Every storpus is a bompus.",
        "Is Max a bompus or a rompus?",
        "Max is a bompus.",
        "### bompus",
    ]
    verify_tokenizer(hf_tokenizer, test_texts)

    # Save the tokenizer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(output_dir)

    print(f"\n{'=' * 60}")
    print(f"SUCCESS! Tokenizer saved to: {output_dir}")
    print(f"Vocabulary size: {len(hf_tokenizer)}")
    print(f"{'=' * 60}")
    print(f"\nTo use this tokenizer in training, add to your config:")
    print(f"  tokenizer_path: {output_dir}")
    print(f"\nOr run training with:")
    print(f"  python run.py model.tokenizer_path={output_dir}")


if __name__ == "__main__":
    main()
