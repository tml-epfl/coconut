# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import datetime
import hashlib
import json
import os
import random
import subprocess
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np
import torch

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore
except ImportError:  # pragma: no cover - OmegaConf is optional at import time
    DictConfig = None  # type: ignore
    ListConfig = None  # type: ignore
    OmegaConf = None  # type: ignore


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _materialize_config(config: Any) -> Any:
    if OmegaConf is not None and isinstance(config, (DictConfig, ListConfig)):
        return OmegaConf.to_container(config, resolve=True)
    return config


def _flatten(
    data: Any,
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    if isinstance(data, Mapping):
        for key in sorted(data.keys()):
            value = data[key]
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
            items.update(_flatten(value, new_key, sep=sep))
    elif isinstance(data, (list, tuple)):
        for index, value in enumerate(data):
            new_key = f"{parent_key}{sep}{index}" if parent_key else str(index)
            items.update(_flatten(value, new_key, sep=sep))
    else:
        items[parent_key] = data
    return items


def canonicalize_config(
    config: Any,
    ignore_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    materialized = _materialize_config(config)
    flattened = _flatten(materialized)
    ignore = set(ignore_keys or [])
    result: Dict[str, Any] = {}
    for key, value in flattened.items():
        if key in ignore:
            continue
        result[key] = value
    return result


def _encode_json(data: Any) -> str:
    def _default(obj: Any) -> Any:
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        return str(obj)

    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        default=_default,
    )


def get_git_metadata(repo_root: Optional[str] = None) -> Dict[str, Any]:
    def _run_git(args: Iterable[str]) -> Tuple[int, str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            return result.returncode, result.stdout.strip()
        except (FileNotFoundError, OSError):  # pragma: no cover - git not available
            return 1, ""

    metadata: Dict[str, Any] = {
        "commit": None,
        "is_dirty": False,
        "diff_sha": None,
        "repo_root": repo_root,
    }

    code, stdout = _run_git(["rev-parse", "--show-toplevel"])
    if code == 0:
        repo_root = stdout
        metadata["repo_root"] = repo_root

    code, commit = _run_git(["rev-parse", "HEAD"])
    if code == 0 and commit:
        metadata["commit"] = commit

    code, status_output = _run_git(["status", "--porcelain"])
    is_dirty = code == 0 and bool(status_output)
    metadata["is_dirty"] = is_dirty

    if is_dirty:
        code, diff_output = _run_git(["diff", "HEAD"])
        if code == 0:
            diff_sha = hashlib.sha256(diff_output.encode("utf-8")).hexdigest()
            metadata["diff_sha"] = diff_sha

    return metadata


def compute_experiment_signature(
    config: Any,
    *,
    ignore_keys: Optional[Iterable[str]] = None,
    git_metadata: Optional[Dict[str, Any]] = None,
    extra: Optional[MutableMapping[str, Any]] = None,
    hash_len: int = 16,
) -> str:
    canonical_config = canonicalize_config(config, ignore_keys=ignore_keys)

    payload: Dict[str, Any] = {
        "config": canonical_config,
        "git": git_metadata or get_git_metadata(),
    }
    if extra:
        payload["extra"] = extra

    serialized = _encode_json(payload)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest[:hash_len]


_BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def generate_attempt_id(length: int = 8) -> str:
    if length <= 0:
        raise ValueError("length must be positive")
    return "".join(random.choice(_BASE62_ALPHABET) for _ in range(length))
