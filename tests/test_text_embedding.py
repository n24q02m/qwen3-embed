from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.text.text_embedding import TextEmbedding
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


class MockEmbedding(TextEmbeddingBase):
    """Mock embedding model for testing delegation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed: Any = MagicMock(return_value=iter([np.zeros(10)]))
        self.query_embed: Any = MagicMock(return_value=iter([np.zeros(10)]))
        self.passage_embed: Any = MagicMock(return_value=iter([np.zeros(10)]))
        self.token_count: Any = MagicMock(return_value=42)

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return [
            DenseModelDescription(
                model="mock-model",
                sources=ModelSource(hf="mock/model"),
                dim=10,
                model_file="model.onnx",
                description="Mock model",
                license="MIT",
                size_in_GB=0.1,
            )
        ]

    def get_embedding_size(self, model_name: str) -> int:
        return 10


def test_init_unsupported_model_raises_value_error():
    """Verify that TextEmbedding raises a ValueError when initialized with an unsupported model."""
    with pytest.raises(ValueError, match="is not supported in TextEmbedding"):
        TextEmbedding(model_name="unsupported-model-name")


def test_list_supported_models():
    """Verify that list_supported_models returns a list of dictionaries with model descriptions."""
    models = TextEmbedding.list_supported_models()
    assert isinstance(models, list)
    assert len(models) > 0

    # Check that each item is a dictionary with expected keys
    for model in models:
        assert isinstance(model, dict)
        assert "model" in model
        assert "dim" in model
        assert "description" in model
        assert "size_in_GB" in model
        assert "sources" in model


class TestTextEmbeddingDelegation:
    """Verify that TextEmbedding correctly delegates calls to its underlying model."""

    @pytest.fixture
    def mock_text_embedding(self):
        """Fixture to provide a TextEmbedding instance with a MockEmbedding model."""
        with patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [MockEmbedding]):
            return TextEmbedding(model_name="mock-model")

    def test_embed_delegation(self, mock_text_embedding):
        """Verify embed() delegates to the underlying model."""
        docs = ["hello", "world"]
        list(mock_text_embedding.embed(docs, batch_size=1, parallel=2, extra="arg"))
        mock_text_embedding.model.embed.assert_called_once_with(docs, 1, 2, extra="arg")

    def test_query_embed_delegation(self, mock_text_embedding):
        """Verify query_embed() delegates to the underlying model."""
        query = "what is the meaning of life?"
        list(mock_text_embedding.query_embed(query, extra="arg"))
        mock_text_embedding.model.query_embed.assert_called_once_with(query, extra="arg")

    def test_passage_embed_delegation(self, mock_text_embedding):
        """Verify passage_embed() delegates to the underlying model."""
        passages = ["passage 1", "passage 2"]
        list(mock_text_embedding.passage_embed(passages, extra="arg"))
        mock_text_embedding.model.passage_embed.assert_called_once_with(passages, extra="arg")

    def test_token_count_delegation(self, mock_text_embedding):
        """Verify token_count() delegates to the underlying model."""
        texts = ["some", "texts"]
        count = mock_text_embedding.token_count(texts, batch_size=10, extra="arg")
        assert count == 42
        mock_text_embedding.model.token_count.assert_called_once_with(
            texts, batch_size=10, extra="arg"
        )
