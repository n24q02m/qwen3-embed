"""Unit tests for PooledEmbedding class."""

from unittest.mock import Mock

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
        list(embedding._post_process_onnx_output(output))


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


def test_pooled_embedding_missing_mask():
    """Missing test for PooledEmbedding missing mask."""
    # Instantiating PooledEmbedding without model loading
    embedding = PooledEmbedding.__new__(PooledEmbedding)

    # Create an output context with no attention mask
    output = Mock()
    output.attention_mask = None
    output.model_output = np.zeros((1, 3, 4), dtype=np.float32)

    # Expect a ValueError with the correct message
    with pytest.raises(
        ValueError, match="attention_mask must be provided for document post-processing"
    ):
        list(embedding._post_process_onnx_output(output))
