"""Tests for GGUF Cross Encoder functionality."""

import sys
from unittest import mock

import pytest

from qwen3_embed.rerank.cross_encoder.gguf_cross_encoder import (
    Qwen3CrossEncoderGGUF,
    _check_llama_cpp,
)


def test_check_llama_cpp_missing():
    """Test that ImportError is raised when llama-cpp-python is missing."""
    with (
        mock.patch.dict(sys.modules, {"llama_cpp": None}),
        pytest.raises(ImportError, match="llama-cpp-python is required"),
    ):
        _check_llama_cpp()


def test_check_llama_cpp_present():
    """Test that no error is raised when llama-cpp-python is present."""
    # Mocking a successful import
    mock_module = mock.Mock()
    with mock.patch.dict(sys.modules, {"llama_cpp": mock_module}):
        try:
            _check_llama_cpp()
        except ImportError:
            pytest.fail("ImportError raised unexpectedly when llama_cpp is mocked as present")


def test_gguf_cross_encoder_init_missing_dependency():
    """Test that Qwen3CrossEncoderGGUF init fails if dependency is missing."""
    with (
        mock.patch.dict(sys.modules, {"llama_cpp": None}),
        pytest.raises(ImportError, match="llama-cpp-python is required"),
    ):
        # We don't need arguments because it should fail before using them
        Qwen3CrossEncoderGGUF()
