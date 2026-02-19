"""Unit tests for PooledNormalizedEmbedding."""

import numpy as np
import pytest

from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.text.pooled_normalized_embedding import PooledNormalizedEmbedding


class DummyModel(PooledNormalizedEmbedding):
    """Dummy model for testing PooledNormalizedEmbedding without instantiation overhead."""

    def __init__(self):
        pass


class TestPooledNormalizedEmbedding:
    """Test PooledNormalizedEmbedding post-processing."""

    def test_post_process_raises_without_attention_mask(self):
        """Should raise ValueError if attention_mask is missing."""
        output = OnnxOutputContext(
            model_output=np.array([[[1.0, 2.0]]]),
            attention_mask=None,
        )

        model = DummyModel()

        with pytest.raises(ValueError, match="attention_mask must be provided"):
            model._post_process_onnx_output(output)

    def test_post_process_success(self):
        """Should return normalized embeddings when attention_mask is provided."""
        # 1 sample, sequence length 2, hidden size 2
        model_output = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        attention_mask = np.array([[1, 1]], dtype=np.int64)

        output = OnnxOutputContext(
            model_output=model_output,
            attention_mask=attention_mask,
        )

        model = DummyModel()

        result = model._post_process_onnx_output(output)

        # Expected behavior:
        # mean_pooling([[[1,2], [3,4]]], [[1,1]]) -> [[2, 3]] (average of [1,2] and [3,4])
        # normalize([[2, 3]]) -> [[2/sqrt(13), 3/sqrt(13)]] -> [[0.5547, 0.83205]]

        expected_mean = np.array([[2.0, 3.0]], dtype=np.float32)
        norm = np.linalg.norm(expected_mean, axis=1, keepdims=True)
        expected_normalized = expected_mean / norm

        np.testing.assert_allclose(result, expected_normalized, atol=1e-6)
        # Verify L2 norm is 1
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0], atol=1e-6)
