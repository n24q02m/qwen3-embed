"""Unit tests for last_token_pool and other utility functions."""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from qwen3_embed.common.utils import (
    check_input_length,
    define_cache_dir,
    iter_batch,
    iter_checked_texts,
    last_token_pool,
    mean_pooling,
    normalize,
)


class TestLastTokenPool:
    """Tests for last-token pooling used by Qwen3 embedding."""

    def test_right_padding(self) -> None:
        """Right-padding: last non-padding token varies per sample."""
        # (batch=2, seq_len=4, hidden_dim=3)
        hidden_states = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [0.0, 0.0, 0.0]],
            ]
        )
        # seq_lens: [2, 3]
        attention_mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.int64)

        result = last_token_pool(hidden_states, attention_mask)

        # Sample 0: last non-pad = index 1 → [4, 5, 6]
        np.testing.assert_array_equal(result[0], [4.0, 5.0, 6.0])
        # Sample 1: last non-pad = index 2 → [13, 14, 15]
        np.testing.assert_array_equal(result[1], [13.0, 14.0, 15.0])

    def test_left_padding(self) -> None:
        """Left-padding: all samples' last token is valid → return [:, -1]."""
        hidden_states = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[0.0, 0.0, 0.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
            ]
        )
        # Left-padding: last position always has mask=1
        attention_mask = np.array([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=np.int64)

        result = last_token_pool(hidden_states, attention_mask)

        # Both samples: return [:, -1]
        np.testing.assert_array_equal(result[0], [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(result[1], [13.0, 14.0, 15.0])

    def test_no_padding(self) -> None:
        """No padding at all: all masks = 1 → last position."""
        hidden_states = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
        attention_mask = np.array([[1, 1, 1]], dtype=np.int64)

        result = last_token_pool(hidden_states, attention_mask)

        np.testing.assert_array_equal(result[0], [5.0, 6.0])

    def test_single_token(self) -> None:
        """Edge case: single token per sample."""
        hidden_states = np.array([[[42.0, 43.0]]])
        attention_mask = np.array([[1]], dtype=np.int64)

        result = last_token_pool(hidden_states, attention_mask)
        np.testing.assert_array_equal(result[0], [42.0, 43.0])

    def test_output_shape(self) -> None:
        """Output shape should be (batch_size, hidden_dim)."""
        batch_size, seq_len, hidden_dim = 5, 10, 64
        hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        result = last_token_pool(hidden_states, attention_mask)
        assert result.shape == (batch_size, hidden_dim)


class TestNormalize:
    """Tests for L2 normalization."""

    def test_unit_norm(self) -> None:
        """Normalised vectors should have unit L2 norm."""
        x = np.array([[3.0, 4.0], [1.0, 0.0]])
        normed = normalize(x)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_zero_vector(self) -> None:
        """Zero vector should remain zero (eps prevents division by zero)."""
        x = np.array([[0.0, 0.0]])
        normed = normalize(x)
        # Should be very small, not NaN
        assert not np.any(np.isnan(normed))


class TestMeanPooling:
    """Tests for mean pooling."""

    def test_with_mask(self) -> None:
        """Mean pooling should only average over non-masked positions."""
        # (batch=1, seq_len=3, hidden_dim=2)
        model_output = np.array([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
        mask = np.array([[1, 1, 0]], dtype=np.int64)

        result = mean_pooling(model_output, mask)
        np.testing.assert_allclose(result[0], [2.0, 3.0])

    def test_full_mask(self) -> None:
        """Full mask = regular average."""
        model_output = np.array([[[2.0, 4.0], [4.0, 8.0]]])
        mask = np.array([[1, 1]], dtype=np.int64)

        result = mean_pooling(model_output, mask)
        np.testing.assert_allclose(result[0], [3.0, 6.0])

    def test_empty_mask(self) -> None:
        """Empty mask should return zero vectors (avoid division by zero)."""
        model_output = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        mask = np.array([[0, 0]], dtype=np.int64)

        result = mean_pooling(model_output, mask)
        np.testing.assert_allclose(result[0], [0.0, 0.0])

    def test_single_token(self) -> None:
        """Single token sequences should work seamlessly."""
        model_output = np.array([[[5.0, 10.0]]])
        mask = np.array([[1]], dtype=np.int64)

        result = mean_pooling(model_output, mask)
        np.testing.assert_allclose(result[0], [5.0, 10.0])

    def test_multi_batch(self) -> None:
        """Multiple batches with varied masking should be handled correctly."""
        model_output = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
                [[2.0, 4.0], [4.0, 8.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        mask = np.array(
            [
                [1, 1, 0],
                [1, 1, 1],
                [0, 0, 0],
            ],
            dtype=np.int64,
        )

        result = mean_pooling(model_output, mask)
        # Batch 0: mean([1, 2], [3, 4]) -> [2, 3]
        np.testing.assert_allclose(result[0], [2.0, 3.0])
        # Batch 1: mean([2, 4], [4, 8], [1, 1]) -> [(2+4+1)/3, (4+8+1)/3] -> [7/3, 13/3] -> [2.333, 4.333]
        np.testing.assert_allclose(result[1], [7 / 3.0, 13 / 3.0])
        # Batch 2: empty mask -> [0, 0]
        np.testing.assert_allclose(result[2], [0.0, 0.0])


class TestIterBatch:
    """Tests for iter_batch utility function."""

    def test_exact_batching(self) -> None:
        """Test when iterable length is exactly divisible by batch size."""
        data = [1, 2, 3, 4, 5, 6]
        result = list(iter_batch(data, 3))
        assert result == [[1, 2, 3], [4, 5, 6]]

    def test_remainder_batching(self) -> None:
        """Test when iterable length is not divisible by batch size."""
        data = [1, 2, 3, 4, 5]
        result = list(iter_batch(data, 3))
        assert result == [[1, 2, 3], [4, 5]]

    def test_batch_larger_than_iterable(self) -> None:
        """Test when batch size is larger than iterable length."""
        data = [1, 2, 3]
        result = list(iter_batch(data, 5))
        assert result == [[1, 2, 3]]

    def test_empty_iterable(self) -> None:
        """Test with empty iterable."""
        data = []
        result = list(iter_batch(data, 3))
        assert result == []

    def test_batch_size_one(self) -> None:
        """Test with batch size of 1."""
        data = [1, 2, 3]
        result = list(iter_batch(data, 1))
        assert result == [[1], [2], [3]]

    def test_generator_input(self) -> None:
        """Test with a generator as input."""
        data = (x for x in range(5))
        result = list(iter_batch(data, 2))
        assert result == [[0, 1], [2, 3], [4]]

    def test_batch_size_zero(self) -> None:
        """Test with batch size 0 (should return empty list)."""
        data = [1, 2, 3]
        result = list(iter_batch(data, 0))
        assert result == []

    def test_negative_batch_size(self) -> None:
        """Test with negative batch size (should raise ValueError)."""
        data = [1, 2, 3]
        with pytest.raises(ValueError, match="Stop argument for islice"):
            list(iter_batch(data, -1))


class TestGetAllPunctuation:
    """Tests for get_all_punctuation utility function."""

    def test_returns_frozenset(self) -> None:
        """Should return a frozenset."""
        from qwen3_embed.common.utils import get_all_punctuation

        result = get_all_punctuation()
        assert isinstance(result, frozenset)

    def test_contains_common_punctuation(self) -> None:
        """Should contain common ASCII punctuation marks."""
        from qwen3_embed.common.utils import get_all_punctuation

        punctuation = get_all_punctuation()

        # Not all of these might be categorized as "P" (Punctuation) in Unicode (e.g., $ is "Sc" Currency Symbol)
        # So let's test a subset that are definitively punctuation
        definite_punctuation = [
            ".",
            ",",
            "!",
            "?",
            "-",
            ";",
            ":",
            "'",
            '"',
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
        ]
        for mark in definite_punctuation:
            assert mark in punctuation, f"Expected {mark} to be in punctuation set"

    def test_does_not_contain_alphanumeric(self) -> None:
        """Should not contain alphanumeric characters."""
        from qwen3_embed.common.utils import get_all_punctuation

        punctuation = get_all_punctuation()
        alphanumeric = ["a", "Z", "0", "9", " ", "\n", "\t"]
        for char in alphanumeric:
            assert char not in punctuation, f"Expected {char} to NOT be in punctuation set"

    def test_caching(self) -> None:
        """Multiple calls should return the exact same object due to lru_cache."""
        from qwen3_embed.common.utils import get_all_punctuation

        result1 = get_all_punctuation()
        result2 = get_all_punctuation()
        assert result1 is result2

    def test_non_ascii_punctuation(self) -> None:
        """Verify non-ASCII punctuation marks are included."""
        from qwen3_embed.common.utils import get_all_punctuation

        punctuation = get_all_punctuation()
        assert "\u00bf" in punctuation  # inverted question mark
        assert "\u00ab" in punctuation  # left double angle quotation mark
        assert "\u2014" in punctuation  # em dash

    def test_excludes_symbols(self) -> None:
        """Verify math and currency symbols are excluded."""
        from qwen3_embed.common.utils import get_all_punctuation

        punctuation = get_all_punctuation()
        assert "+" not in punctuation
        assert "=" not in punctuation
        assert "$" not in punctuation
        assert "\u20ac" not in punctuation  # euro sign


class TestDefineCacheDir:
    @patch("qwen3_embed.common.utils.Path.mkdir")
    @patch("qwen3_embed.common.utils.Path.chmod")
    def test_custom_argument(self, mock_chmod, mock_mkdir):
        res = define_cache_dir("/custom/arg/path")
        assert res == Path("/custom/arg/path")
        mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)
        mock_chmod.assert_called_once_with(0o700)

    @patch("qwen3_embed.common.utils.Path.mkdir")
    @patch("qwen3_embed.common.utils.Path.chmod")
    @patch.dict(os.environ, {"QWEN3_EMBED_CACHE_PATH": "/custom/env/path"})
    def test_qwen3_cache_path(self, mock_chmod, mock_mkdir):
        res = define_cache_dir()
        assert res == Path("/custom/env/path")
        mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)
        mock_chmod.assert_called_once_with(0o700)

    @patch("qwen3_embed.common.utils.Path.mkdir")
    @patch("qwen3_embed.common.utils.Path.chmod")
    @patch.dict(os.environ, {"XDG_CACHE_HOME": "/xdg/cache"}, clear=True)
    def test_xdg_cache_home(self, mock_chmod, mock_mkdir):
        res = define_cache_dir()
        assert res == Path("/xdg/cache/qwen3_embed")
        mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)
        mock_chmod.assert_called_once_with(0o700)

    @patch("qwen3_embed.common.utils.Path.mkdir")
    @patch("qwen3_embed.common.utils.Path.chmod")
    def test_fallback_home_cache(self, mock_chmod, mock_mkdir):
        # Resolve the expected home dir BEFORE clearing the environment, since
        # ``Path.home()`` on Windows depends on env vars (USERPROFILE etc.) that
        # ``clear=True`` would wipe and cause a RuntimeError.
        fake_home = Path.home()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("qwen3_embed.common.utils.Path.home", return_value=fake_home),
        ):
            res = define_cache_dir()
        assert res == fake_home / ".cache/qwen3_embed"
        mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)
        mock_chmod.assert_called_once_with(0o700)

    @patch("qwen3_embed.common.utils.Path.mkdir")
    @patch("qwen3_embed.common.utils.Path.chmod", side_effect=OSError("chmod failed"))
    def test_chmod_oserror_suppressed(self, mock_chmod, mock_mkdir):
        res = define_cache_dir("/custom/arg/path")
        assert res == Path("/custom/arg/path")
        mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)
        mock_chmod.assert_called_once_with(0o700)


class TestInputValidation:
    """Tests for input length validation utilities."""

    def test_check_input_length_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should not raise error when within limits."""
        import qwen3_embed.common.utils

        monkeypatch.setattr(qwen3_embed.common.utils, "MAX_INPUT_LENGTH", 5)

        # Exact limit
        check_input_length("abcde")
        # Below limit
        check_input_length("abcd")

    def test_check_input_length_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when limit is exceeded."""
        import qwen3_embed.common.utils

        monkeypatch.setattr(qwen3_embed.common.utils, "MAX_INPUT_LENGTH", 5)

        with pytest.raises(
            ValueError, match="Input string exceeds maximum allowed length of 5 characters"
        ):
            check_input_length("abcdef")

    def test_iter_checked_texts_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generator should yield all texts if they are within limits."""
        import qwen3_embed.common.utils

        monkeypatch.setattr(qwen3_embed.common.utils, "MAX_INPUT_LENGTH", 5)

        texts = ["abc", "de", "fghij"]
        result = list(iter_checked_texts(texts))
        assert result == texts

    def test_iter_checked_texts_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generator should yield valid texts before raising ValueError on invalid one."""
        import qwen3_embed.common.utils

        monkeypatch.setattr(qwen3_embed.common.utils, "MAX_INPUT_LENGTH", 5)

        texts = ["abc", "abcdef", "valid"]
        iterator = iter_checked_texts(texts)

        assert next(iterator) == "abc"
        with pytest.raises(
            ValueError, match="Input string exceeds maximum allowed length of 5 characters"
        ):
            next(iterator)
