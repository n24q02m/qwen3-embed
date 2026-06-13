"""Tests for GGUF Text Embedding (Qwen3TextEmbeddingGGUF)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock llama_cpp before importing the module under test
_mock_llama_module = MagicMock()
sys.modules["llama_cpp"] = _mock_llama_module

from qwen3_embed.common.utils import _check_llama_cpp  # noqa: E402
from qwen3_embed.text.gguf_embedding import (  # noqa: E402
    DEFAULT_TASK,
    QUERY_INSTRUCTION_TEMPLATE,
    Qwen3TextEmbeddingGGUF,
    supported_qwen3_gguf_models,
)

# ---------------------------------------------------------------------------
# Tests for _check_llama_cpp
# ---------------------------------------------------------------------------


def test_check_llama_cpp_missing():
    """Test that _check_llama_cpp raises ImportError when llama_cpp is absent."""
    with (
        patch.dict(sys.modules, {"llama_cpp": None}),
        pytest.raises(ImportError) as excinfo,
    ):
        _check_llama_cpp()

    # Verify message and exception chaining (raise ... from e)
    assert "llama-cpp-python is required for GGUF models" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_check_llama_cpp_import_error_with_mock():
    """Test that _check_llama_cpp raises ImportError from builtins.__import__."""
    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("Mocked import error for llama_cpp")
        return original_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        pytest.raises(ImportError) as excinfo,
    ):
        _check_llama_cpp()

    assert "llama-cpp-python is required for GGUF models" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None
    assert "Mocked import error for llama_cpp" in str(excinfo.value.__cause__)


def test_check_llama_cpp_exact_message():
    """Test that _check_llama_cpp raises ImportError with the exact expected message."""
    expected_msg = (
        "llama-cpp-python is required for GGUF models. Install with: pip install qwen3-embed[gguf]"
    )
    with (
        patch.dict(sys.modules, {"llama_cpp": None}),
        pytest.raises(ImportError) as excinfo,
    ):
        _check_llama_cpp()

    assert str(excinfo.value) == expected_msg


def test_check_llama_cpp_present():
    """Test that _check_llama_cpp does not raise when llama_cpp is present."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"llama_cpp": mock_module}):
        _check_llama_cpp()  # Should not raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_gguf_model():
    """Fixture to mock the internal Llama model."""
    with patch("llama_cpp.Llama") as mock_l:
        instance = mock_l.return_value

        def mock_create_embedding(texts):
            return {"data": [{"embedding": [0.1] * 1024} for _ in range(len(texts))]}

        instance.create_embedding.side_effect = mock_create_embedding
        yield instance


# ---------------------------------------------------------------------------
# Core Logic Tests
# ---------------------------------------------------------------------------


def test_gguf_embedding_init_calls_check(mock_gguf_model):
    """Test that Qwen3TextEmbeddingGGUF calls _check_llama_cpp on init."""
    with (
        patch("qwen3_embed.text.gguf_embedding._check_llama_cpp") as mock_check,
        patch.object(Qwen3TextEmbeddingGGUF, "_get_model_description"),
        patch("qwen3_embed.text.gguf_embedding.define_cache_dir"),
        patch.object(Qwen3TextEmbeddingGGUF, "download_model"),
        patch("qwen3_embed.text.gguf_embedding.Path.exists", return_value=True),
    ):
        Qwen3TextEmbeddingGGUF()
        mock_check.assert_called_once()


def test_embed_basic(mock_gguf_model):
    """Test basic embedding functionality."""
    with (
        patch("qwen3_embed.text.gguf_embedding._check_llama_cpp"),
        patch.object(Qwen3TextEmbeddingGGUF, "_get_model_description"),
        patch("qwen3_embed.text.gguf_embedding.define_cache_dir"),
        patch.object(Qwen3TextEmbeddingGGUF, "download_model"),
        patch("qwen3_embed.text.gguf_embedding.Path.exists", return_value=True),
    ):
        model = Qwen3TextEmbeddingGGUF()
        model._llm = mock_gguf_model

        docs = ["hello", "world"]
        embeddings = list(model.embed(docs))

        assert len(embeddings) == 2
        assert isinstance(embeddings[0], np.ndarray)
        assert embeddings[0].shape == (1024,)
        # Verify L2 normalization (sum of squares should be approx 1.0)
        assert np.linalg.norm(embeddings[0]) == pytest.approx(1.0)


def test_embed_batching(mock_gguf_model):
    """Test that batch_size is respected (though passed to llama-cpp-python)."""
    with (
        patch("qwen3_embed.text.gguf_embedding._check_llama_cpp"),
        patch.object(Qwen3TextEmbeddingGGUF, "_get_model_description"),
        patch("qwen3_embed.text.gguf_embedding.define_cache_dir"),
        patch.object(Qwen3TextEmbeddingGGUF, "download_model"),
        patch("qwen3_embed.text.gguf_embedding.Path.exists", return_value=True),
    ):
        model = Qwen3TextEmbeddingGGUF()
        model._llm = mock_gguf_model

        docs = ["doc1", "doc2", "doc3"]
        list(model.embed(docs, batch_size=2))

        # Should be called twice (one batch of 2, one batch of 1)
        assert mock_gguf_model.create_embedding.call_count == 2


def test_embed_mrl(mock_gguf_model):
    """Test MRL truncation."""
    with (
        patch("qwen3_embed.text.gguf_embedding._check_llama_cpp"),
        patch.object(Qwen3TextEmbeddingGGUF, "_get_model_description"),
        patch("qwen3_embed.text.gguf_embedding.define_cache_dir"),
        patch.object(Qwen3TextEmbeddingGGUF, "download_model"),
        patch("qwen3_embed.text.gguf_embedding.Path.exists", return_value=True),
    ):
        model = Qwen3TextEmbeddingGGUF()
        model._llm = mock_gguf_model

        embeddings = list(model.embed("hello", dim=256))
        assert embeddings[0].shape == (256,)
        assert np.linalg.norm(embeddings[0]) == pytest.approx(1.0)


def test_query_embed(mock_gguf_model):
    """Test query embedding with instruction prefix."""
    with (
        patch("qwen3_embed.text.gguf_embedding._check_llama_cpp"),
        patch.object(Qwen3TextEmbeddingGGUF, "_get_model_description"),
        patch("qwen3_embed.text.gguf_embedding.define_cache_dir"),
        patch.object(Qwen3TextEmbeddingGGUF, "download_model"),
        patch("qwen3_embed.text.gguf_embedding.Path.exists", return_value=True),
    ):
        model = Qwen3TextEmbeddingGGUF()
        model._llm = mock_gguf_model

        query = "What is AI?"
        with patch.object(model, "embed", wraps=model.embed) as mock_embed:
            list(model.query_embed(query))

            # Verify instruction template was applied
            expected_text = QUERY_INSTRUCTION_TEMPLATE.format(task=DEFAULT_TASK, text=query)
            # The first argument to embed is the documents list/iterable
            # We convert to list for comparison if it was a generator
            called_docs = list(mock_embed.call_args[0][0])
            assert called_docs == [expected_text]


def test_query_embed_custom_task(mock_gguf_model):
    """Test query embedding with custom task instruction."""
    with (
        patch("qwen3_embed.text.gguf_embedding._check_llama_cpp"),
        patch.object(Qwen3TextEmbeddingGGUF, "_get_model_description"),
        patch("qwen3_embed.text.gguf_embedding.define_cache_dir"),
        patch.object(Qwen3TextEmbeddingGGUF, "download_model"),
        patch("qwen3_embed.text.gguf_embedding.Path.exists", return_value=True),
    ):
        model = Qwen3TextEmbeddingGGUF()
        model._llm = mock_gguf_model

        query = "What is AI?"
        custom_task = "Custom retrieve"
        with patch.object(model, "embed", wraps=model.embed) as mock_embed:
            list(model.query_embed(query, task=custom_task))

            expected_text = QUERY_INSTRUCTION_TEMPLATE.format(task=custom_task, text=query)
            called_docs = list(mock_embed.call_args[0][0])
            assert called_docs == [expected_text]


def test_passage_embed(mock_gguf_model):
    """Test passage embedding (should NOT have instruction)."""
    with (
        patch("qwen3_embed.text.gguf_embedding._check_llama_cpp"),
        patch.object(Qwen3TextEmbeddingGGUF, "_get_model_description"),
        patch("qwen3_embed.text.gguf_embedding.define_cache_dir"),
        patch.object(Qwen3TextEmbeddingGGUF, "download_model"),
        patch("qwen3_embed.text.gguf_embedding.Path.exists", return_value=True),
    ):
        model = Qwen3TextEmbeddingGGUF()
        model._llm = mock_gguf_model

        text = "AI is cool."
        with patch.object(model, "embed", wraps=model.embed) as mock_embed:
            list(model.passage_embed([text]))

            called_docs = list(mock_embed.call_args[0][0])
            assert called_docs == [text]


class TestGGUFEmbeddingExtra:
    def test_list_supported_models(self):
        """Test _list_supported_models returns the supported models list."""
        assert Qwen3TextEmbeddingGGUF._list_supported_models() == supported_qwen3_gguf_models
