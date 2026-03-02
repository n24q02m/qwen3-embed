"""Unit tests for PooledEmbedding class."""

import numpy as np
import pytest

from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.text.pooled_embedding import PooledEmbedding


class MockPooledEmbedding(PooledEmbedding):
    """Mock class to bypass model loading."""

    def __init__(self):
        pass


def test_post_process_raises_without_attention_mask():
    """Test that _post_process_onnx_output raises ValueError if attention_mask is missing."""
    embedding = MockPooledEmbedding()
    output = OnnxOutputContext(
        model_output=np.zeros((1, 3, 4), dtype=np.float32), attention_mask=None
    )
    with pytest.raises(
        ValueError, match="attention_mask must be provided for document post-processing"
    ):
        embedding._post_process_onnx_output(output)


def test_post_process_mean_pooling_works():
    """Test that _post_process_onnx_output correctly performs mean pooling."""
    embedding = MockPooledEmbedding()
    # Batch 1, Seq 2, Dim 2
    model_output = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    attention_mask = np.array([[1, 1]], dtype=np.int64)
    output = OnnxOutputContext(model_output=model_output, attention_mask=attention_mask)

    result = list(embedding._post_process_onnx_output(output))
    # Mean of [[1, 2], [3, 4]] is [2, 3]
    expected = np.array([[2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)

def test_get_worker_class():
    """Test that _get_worker_class returns PooledEmbeddingWorker."""
    from qwen3_embed.text.pooled_embedding import PooledEmbeddingWorker
    assert PooledEmbedding._get_worker_class() is PooledEmbeddingWorker

def test_list_supported_models():
    """Test that _list_supported_models returns supported_pooled_models."""
    from qwen3_embed.text.pooled_embedding import supported_pooled_models
    assert PooledEmbedding._list_supported_models() is supported_pooled_models

def test_init_embedding(monkeypatch):
    """Test that PooledEmbeddingWorker.init_embedding returns a PooledEmbedding."""
    from qwen3_embed.text.pooled_embedding import PooledEmbeddingWorker

    # Mock __init__ of PooledEmbedding to bypass ONNX model loading
    monkeypatch.setattr(PooledEmbedding, "__init__", lambda *args, **kwargs: None)

    monkeypatch.setattr(PooledEmbeddingWorker, "__init__", lambda *args, **kwargs: None)
    worker = PooledEmbeddingWorker()
    embedding = worker.init_embedding(model_name="test_model", cache_dir="/test/cache", some_kwarg="value")

    assert isinstance(embedding, PooledEmbedding)

def test_pooled_embedding_missing_mask():
    """Missing test for PooledEmbedding missing mask."""
    # Instantiating PooledEmbedding without model loading
    embedding = PooledEmbedding.__new__(PooledEmbedding)

    # Create an output context with no attention mask
    output = OnnxOutputContext(
        model_output=np.zeros((1, 3, 4), dtype=np.float32),
        attention_mask=None
    )

    # Expect a ValueError with the correct message
    with pytest.raises(
        ValueError, match="attention_mask must be provided for document post-processing"
    ):
        embedding._post_process_onnx_output(output)
