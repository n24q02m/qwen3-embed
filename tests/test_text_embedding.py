from collections.abc import Iterable
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.text.text_embedding import TextEmbedding
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


class MockEmbedding(TextEmbeddingBase):
    """Mock implementation of TextEmbeddingBase for testing delegation."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.mock_embed = MagicMock(return_value=iter([np.zeros(10)]))
        self.mock_query_embed = MagicMock(return_value=iter([np.zeros(10)]))
        self.mock_passage_embed = MagicMock(return_value=iter([np.zeros(10)]))
        self.mock_token_count = MagicMock(return_value=42)

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return [
            DenseModelDescription(
                model="mock/test-model",
                sources=ModelSource(hf="mock/test-model"),
                dim=10,
                model_file="model.onnx",
                description="Mock model for testing",
                license="MIT",
                size_in_GB=0.1,
            )
        ]

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[np.ndarray]:
        return self.mock_embed(documents, batch_size, parallel, **kwargs)

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[np.ndarray]:
        return self.mock_query_embed(query, **kwargs)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[np.ndarray]:
        return self.mock_passage_embed(texts, **kwargs)

    def token_count(self, texts: str | Iterable[str], batch_size: int = 1024, **kwargs: Any) -> int:
        return self.mock_token_count(texts, batch_size=batch_size, **kwargs)


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


def test_delegation():
    """Verify that TextEmbedding correctly delegates calls to its underlying model."""
    original_registry = TextEmbedding.EMBEDDINGS_REGISTRY
    try:
        TextEmbedding.EMBEDDINGS_REGISTRY = [MockEmbedding]

        te = TextEmbedding(model_name="mock/test-model")
        mock_model = cast(MockEmbedding, te.model)

        # Test embed
        docs = ["doc1", "doc2"]
        list(te.embed(docs, batch_size=128))
        mock_model.mock_embed.assert_called_once_with(docs, 128, None)

        # Test query_embed
        query = "test query"
        list(te.query_embed(query, some_arg="value"))
        mock_model.mock_query_embed.assert_called_once_with(query, some_arg="value")

        # Test passage_embed
        passages = ["passage1", "passage2"]
        list(te.passage_embed(passages, another_arg="foo"))
        mock_model.mock_passage_embed.assert_called_once_with(passages, another_arg="foo")

        # Test token_count
        texts = ["text1", "text2"]
        count = te.token_count(texts, batch_size=512)
        assert count == 42
        mock_model.mock_token_count.assert_called_once_with(texts, batch_size=512)

    finally:
        TextEmbedding.EMBEDDINGS_REGISTRY = original_registry
