"""Unit tests for GGUF embedding functionality."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from qwen3_embed.text.gguf_embedding import _check_llama_cpp


class TestGGUFDependencyCheck:
    """Test the GGUF dependency check."""

    def test_check_llama_cpp_installed(self):
        """Should pass if llama-cpp-python is importable."""
        # Use patch.dict to safely modify sys.modules during the test
        with patch.dict(sys.modules, {"llama_cpp": MagicMock()}):
            # Should not raise
            _check_llama_cpp()

    def test_check_llama_cpp_missing(self):
        """Should raise ImportError if llama-cpp-python is missing."""
        # Simulate missing module by setting it to None in sys.modules
        # This causes imports of that module to fail with ImportError (or ModuleNotFoundError)
        with (
            patch.dict(sys.modules, {"llama_cpp": None}),
            pytest.raises(ImportError, match="llama-cpp-python is required"),
        ):
            _check_llama_cpp()
