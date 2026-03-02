import contextlib
import functools
import os
import re
import sys
import unicodedata
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from qwen3_embed.common.types import NumpyArray

T = TypeVar("T")


def last_token_pool(input_array: NumpyArray, attention_mask: NDArray[np.int64]) -> NumpyArray:
    """Extract embedding from the last non-padding token position.

    Qwen3-Embedding uses last-token pooling (NOT CLS/mean pooling).
    Handles both left-padding and right-padding.

    Args:
        input_array: Model output, shape (batch_size, seq_len, hidden_dim).
        attention_mask: Attention mask, shape (batch_size, seq_len).

    Returns:
        Pooled embeddings, shape (batch_size, hidden_dim).
    """
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return input_array[:, -1]

    batch_size, seq_len = attention_mask.shape
    # Find the index of the last '1' in the attention mask for each row
    # argmax returns the *first* occurrence of the max value.
    # By reversing the mask, we find the first '1' from the end.
    last_token_indices = seq_len - 1 - np.argmax(attention_mask[:, ::-1], axis=1)
    return input_array[np.arange(batch_size), last_token_indices]


def normalize(input_array: NumpyArray, p: int = 2, dim: int = 1, eps: float = 1e-12) -> NumpyArray:
    # Calculate the Lp norm along the specified dimension
    norm = np.linalg.norm(input_array, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    normalized_array = input_array / norm
    return normalized_array


def mean_pooling(input_array: NumpyArray, attention_mask: NDArray[np.int64]) -> NumpyArray:
    # Use broadcasting instead of np.tile, and cast mask to input dtype to avoid type promotion overhead
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(input_array.dtype)
    sum_embeddings = np.sum(input_array * input_mask_expanded, axis=1)
    sum_mask = np.sum(input_mask_expanded, axis=1)
    pooled_embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)
    return pooled_embeddings


def iter_batch[T](iterable: Iterable[T], size: int) -> Iterable[list[T]]:
    """
    >>> list(iter_batch([1,2,3,4,5], 3))
    [[1, 2, 3], [4, 5]]
    """
    source_iter = iter(iterable)
    while source_iter:
        b = list(islice(source_iter, size))
        if len(b) == 0:
            break
        yield b


def define_cache_dir(cache_dir: str | None = None) -> Path:
    """
    Define the cache directory for qwen3_embed
    """
    if cache_dir is None:
        if os.environ.get("QWEN3_EMBED_CACHE_PATH"):
            cache_path = Path(os.environ["QWEN3_EMBED_CACHE_PATH"])
        else:
            xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
            base_path = Path(xdg_cache_home) if xdg_cache_home else Path.home() / ".cache"
            cache_path = base_path / "qwen3_embed"
    else:
        cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    with contextlib.suppress(OSError):
        cache_path.chmod(0o700)

    return cache_path


@functools.lru_cache
def get_all_punctuation() -> set[str]:
    return set(
        chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
    )


def remove_non_alphanumeric(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
