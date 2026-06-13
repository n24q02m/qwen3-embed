"""Tests for CustomTextEmbedding — covers custom_text_embedding.py."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from qwen3_embed.common.model_description import (
    CustomDenseModelDescription,
    DenseModelDescription,
    ModelSource,
    PoolingType,
)
from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.text.custom_text_embedding import CustomTextEmbedding

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "test-org/custom-embed-model"
_MODEL_DESC = DenseModelDescription(
    model=_MODEL_NAME,
    sources=ModelSource(hf=_MODEL_NAME),
    model_file="onnx/model.onnx",
    description="test custom embedding",
    license="MIT",
    size_in_GB=0.1,
    dim=4,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embeddings(shape: tuple[int, ...], dtype=np.float32) -> NumpyArray:
    return np.ones(shape, dtype=dtype)


def _make_attention_mask(batch: int, seq_len: int, pad_from: int = -1) -> np.ndarray:
    """Create attention mask with 1s up to pad_from and 0s after.

    If pad_from is -1 (default), all tokens are marked as real (full mask).
    """
    if pad_from == -1:
        pad_from = seq_len
    mask = np.zeros((batch, seq_len), dtype=np.int64)
    mask[:, :pad_from] = 1
    return mask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    """Isolate the class-level registry between tests."""
    original = dict(CustomTextEmbedding._SUPPORTED)
    yield
    CustomTextEmbedding._SUPPORTED.clear()
    CustomTextEmbedding._SUPPORTED.update(original)


def _register(
    model_name: str = _MODEL_NAME,
    pooling: PoolingType = PoolingType.CLS,
    normalization: bool = True,
) -> CustomDenseModelDescription:
    desc = CustomDenseModelDescription(
        model=model_name,
        sources=ModelSource(hf=model_name),
        model_file="onnx/model.onnx",
        description="test",
        license="MIT",
        size_in_GB=0.1,
        dim=4,
        pooling=pooling,
        normalization=normalization,
    )
    CustomTextEmbedding._register(desc)
    return desc


def _build(
    tmp_path: Path,
    model_name: str = _MODEL_NAME,
    pooling: PoolingType = PoolingType.CLS,
    normalization: bool = True,
    **kwargs: Any,
) -> CustomTextEmbedding:
    """Build a CustomTextEmbedding with download + load mocked out."""
    _register(model_name=model_name, pooling=pooling, normalization=normalization)
    with (
        patch.object(CustomTextEmbedding, "download_model", return_value=tmp_path),
        patch.object(CustomTextEmbedding, "load_onnx_model"),
    ):
        return CustomTextEmbedding(model_name=model_name, lazy_load=False, **kwargs)


# =+
# CustomTextEmbedding._post_process_onnx_output (line 64)
# ===========================================================================


class TestPostProcessOnnxOutput:
    def _output(self, batch: int, seq_len: int, dim: int) -> OnnxOutputContext:
        embeddings = np.ones((batch, seq_len, dim), dtype=np.float32)
        mask = _make_attention_mask(batch=batch, seq_len=seq_len, pad_from=seq_len)
        return OnnxOutputContext(model_output=embeddings, attention_mask=mask)

    def test_cls_pooling_normalized(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.CLS, normalization=True)
        ctx = self._output(2, 5, 4)
        result = np.asarray(emb._post_process_onnx_output(ctx))
        # shape should be (2, 4) — CLS token from each row, normalized
        assert result.shape == (2, 4)
        # Each row should be normalized to unit length
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)

    def test_mean_pooling_with_mask(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.MEAN, normalization=False)
        ctx = self._output(2, 4, 8)
        result = np.asarray(emb._post_process_onnx_output(ctx))
        assert result.shape == (2, 8)

    def test_last_token_pooling(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.LAST_TOKEN, normalization=False)
        ctx = self._output(1, 4, 6)
        result = np.asarray(emb._post_process_onnx_output(ctx))
        assert result.shape == (1, 6)

    def test_disabled_pooling(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.DISABLED, normalization=False)
        embeddings = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        mask = _make_attention_mask(2, 3)
        ctx = OnnxOutputContext(model_output=embeddings, attention_mask=mask)
        result = np.asarray(emb._post_process_onnx_output(ctx))
        np.testing.assert_array_equal(result, embeddings)

    def test_normalization_applied_after_pool(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.CLS, normalization=True)
        # Non-unit-norm embeddings at CLS position
        embeddings = np.array([[[3.0, 4.0, 0.0, 0.0]]], dtype=np.float32)  # (1, 1, 4)
        mask = np.ones((1, 1), dtype=np.int64)
        ctx = OnnxOutputContext(model_output=embeddings, attention_mask=mask)
        result = np.asarray(emb._post_process_onnx_output(ctx))
        # After norm, L2 should be 1.0
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_no_normalization(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.CLS, normalization=False)
        embeddings = np.array([[[3.0, 4.0, 0.0, 0.0]]], dtype=np.float32)  # (1, 1, 4)
        mask = np.ones((1, 1), dtype=np.int64)
        ctx = OnnxOutputContext(model_output=embeddings, attention_mask=mask)
        result = np.asarray(emb._post_process_onnx_output(ctx))
        # Unnormalized: L2 should be 5.0
        np.testing.assert_allclose(np.linalg.norm(result), 5.0, atol=1e-6)
