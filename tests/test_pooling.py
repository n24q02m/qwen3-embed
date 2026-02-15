"""Unit tests for last_token_pool, PoolingType.LAST_TOKEN, and utility functions."""

import numpy as np

from qwen3_embed.common.model_description import PoolingType
from qwen3_embed.common.utils import last_token_pool, normalize


class TestLastTokenPool:
    """Test last_token_pool with various padding scenarios."""

    def test_last_token_pool_right_padding_returns_last_valid_token(self):
        """Right-padding: last non-pad token varies per sample."""
        hidden = np.array(
            [
                [[1, 2], [3, 4], [0, 0]],  # seq_len=2 (pad at pos 2)
                [[5, 6], [7, 8], [9, 10]],  # seq_len=3
            ],
            dtype=np.float32,
        )
        mask = np.array(
            [
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=np.int64,
        )
        result = last_token_pool(hidden, mask)
        # Sample 0: last valid = index 1 -> [3, 4]
        # Sample 1: last valid = index 2 -> [9, 10]
        np.testing.assert_array_equal(result, [[3, 4], [9, 10]])

    def test_last_token_pool_left_padding_returns_last_token(self):
        """Left-padding: all samples end at the last position."""
        hidden = np.array(
            [
                [[0, 0], [1, 2], [3, 4]],  # pad at pos 0
                [[5, 6], [7, 8], [9, 10]],  # no padding
            ],
            dtype=np.float32,
        )
        mask = np.array(
            [
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.int64,
        )
        result = last_token_pool(hidden, mask)
        # Left-padding: last column is all-1 -> use hidden[:, -1]
        np.testing.assert_array_equal(result, [[3, 4], [9, 10]])

    def test_last_token_pool_single_token_returns_token(self):
        """Single-token sequence (edge case)."""
        hidden = np.array([[[42, 43]]], dtype=np.float32)
        mask = np.array([[1]], dtype=np.int64)
        result = last_token_pool(hidden, mask)
        np.testing.assert_array_equal(result, [[42, 43]])

    def test_last_token_pool_batch_size_one_returns_correct_token(self):
        """Batch of size 1 with right-padding."""
        hidden = np.array([[[1, 0], [2, 0], [0, 0]]], dtype=np.float32)
        mask = np.array([[1, 1, 0]], dtype=np.int64)
        result = last_token_pool(hidden, mask)
        np.testing.assert_array_equal(result, [[2, 0]])


class TestPoolingType:
    """Verify PoolingType enum includes LAST_TOKEN."""

    def test_pooling_type_has_last_token_member_returns_true(self):
        assert hasattr(PoolingType, "LAST_TOKEN")
        assert PoolingType.LAST_TOKEN == "LAST_TOKEN"

    def test_pooling_type_enumeration_returns_expected_set(self):
        expected = {"CLS", "MEAN", "LAST_TOKEN", "DISABLED"}
        actual = {pt.value for pt in PoolingType}
        assert actual == expected


class TestNormalize:
    """Sanity check for L2 normalize."""

    def test_normalize_unit_vectors_returns_normalized_vectors(self):
        x = np.array([[3.0, 4.0]], dtype=np.float32)
        result = normalize(x)
        np.testing.assert_allclose(result, [[0.6, 0.8]], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0], atol=1e-6)

    def test_normalize_zero_vector_returns_zero_vector(self):
        """Zero vector should not raise (eps prevents division by zero)."""
        x = np.array([[0.0, 0.0]], dtype=np.float32)
        result = normalize(x)
        assert result.shape == (1, 2)
