"""Unit tests for last_token_pool and other utility functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qwen3_embed.common.utils import define_cache_dir, last_token_pool, mean_pooling, normalize


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


class TestDefineCacheDir:
    """Tests for define_cache_dir."""

    def test_default_behavior(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should use default temp dir if no env var or arg is provided."""
        monkeypatch.delenv("QWEN3_EMBED_CACHE_PATH", raising=False)

        path = define_cache_dir()

        expected_default = Path(tempfile.gettempdir()) / "qwen3_embed_cache"
        assert path == expected_default
        assert path.exists()
        assert path.is_dir()

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Environment variable QWEN3_EMBED_CACHE_PATH should override default."""
        custom_cache = tmp_path / "env_cache"
        monkeypatch.setenv("QWEN3_EMBED_CACHE_PATH", str(custom_cache))

        path = define_cache_dir()

        assert path == custom_cache
        assert path.exists()
        assert path.is_dir()

    def test_explicit_argument(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Explicit argument should define cache directory."""
        # Ensure env var doesn't interfere
        monkeypatch.delenv("QWEN3_EMBED_CACHE_PATH", raising=False)

        explicit_cache = tmp_path / "explicit_cache"
        path = define_cache_dir(str(explicit_cache))

        assert path == explicit_cache
        assert path.exists()
        assert path.is_dir()

    def test_arg_overrides_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Explicit argument should take precedence over environment variable."""
        env_cache = tmp_path / "env_cache"
        explicit_cache = tmp_path / "arg_cache"

        monkeypatch.setenv("QWEN3_EMBED_CACHE_PATH", str(env_cache))

        path = define_cache_dir(str(explicit_cache))

        assert path == explicit_cache
        assert path.exists()
        # Verify that the env path was NOT created by this call
        # (It shouldn't be created unless something else created it)
        assert not env_cache.exists()
