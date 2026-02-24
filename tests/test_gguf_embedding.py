import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock llama_cpp before importing the module under test
sys.modules["llama_cpp"] = MagicMock()

from qwen3_embed.text.gguf_embedding import (  # noqa: E402
    DEFAULT_TASK,
    QUERY_INSTRUCTION_TEMPLATE,
    Qwen3TextEmbeddingGGUF,
)


class TestGGUFEmbeddingQueryEmbed:
    @pytest.fixture
    def model(self):
        """Create a GGUF model instance with mocked dependencies."""
        with patch(
            "qwen3_embed.text.gguf_embedding.Qwen3TextEmbeddingGGUF.__init__", return_value=None
        ):
            model = Qwen3TextEmbeddingGGUF()
            # Mock the embed method since query_embed calls it
            model.embed = MagicMock()
            return model

    def test_query_embed_single_string_default_task(self, model):
        """Test query_embed with a single string and default task."""
        query = "test query"
        list(model.query_embed(query))

        expected_text = QUERY_INSTRUCTION_TEMPLATE.format(task=DEFAULT_TASK, text=query)
        model.embed.assert_called_once()
        call_args = model.embed.call_args
        # embed receives a list of strings even for a single query
        assert call_args[0][0] == [expected_text]

    def test_query_embed_list_default_task(self, model):
        """Test query_embed with a list of strings and default task."""
        queries = ["query1", "query2"]
        list(model.query_embed(queries))

        expected_texts = [
            QUERY_INSTRUCTION_TEMPLATE.format(task=DEFAULT_TASK, text=q) for q in queries
        ]
        model.embed.assert_called_once()
        call_args = model.embed.call_args
        assert call_args[0][0] == expected_texts

    def test_query_embed_custom_task(self, model):
        """Test query_embed with a custom task."""
        query = "test query"
        custom_task = "Custom retrieval task"
        list(model.query_embed(query, task=custom_task))

        expected_text = QUERY_INSTRUCTION_TEMPLATE.format(task=custom_task, text=query)
        model.embed.assert_called_once()
        call_args = model.embed.call_args
        assert call_args[0][0] == [expected_text]
        # Ensure task is popped from kwargs and not passed to embed
        assert "task" not in call_args[1]

    def test_query_embed_passes_kwargs(self, model):
        """Test query_embed passes other kwargs (like dim) to embed."""
        query = "test query"
        list(model.query_embed(query, dim=128, other_arg="value"))

        model.embed.assert_called_once()
        call_args = model.embed.call_args
        assert call_args[1]["dim"] == 128
        assert call_args[1]["other_arg"] == "value"
