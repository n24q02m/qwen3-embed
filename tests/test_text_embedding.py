from collections.abc import Iterable
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.text.text_embedding import TextEmbedding
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


class MockEmbedding(TextEmbeddingBase):
    """Mock implementation of TextEmbeddingBase for testing delegation."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Initialize mocks as attributes so they can be accessed after instantiation
        self.embed_mock = MagicMock(return_value=iter([np.zeros(4)]))
        self.query_embed_mock = MagicMock(return_value=iter([np.zeros(4)]))
        self.passage_embed_mock = MagicMock(return_value=iter([np.zeros(4)]))
        self.token_count_mock = MagicMock(return_value=42)

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return [
            DenseModelDescription(
                model="mock-model",
                sources=ModelSource(hf="mock-model"),
                model_file="model.onnx",
                description="mock",
                license="MIT",
                size_in_GB=0.1,
                dim=4,
            )
        ]

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        return self.embed_mock(documents, batch_size=batch_size, parallel=parallel, **kwargs)

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        return self.query_embed_mock(query, **kwargs)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        return self.passage_embed_mock(texts, **kwargs)

    def token_count(
        self, texts: str | Iterable[str], batch_size: int = 1024, **kwargs: Any
    ) -> int:
        return self.token_count_mock(texts, batch_size=batch_size, **kwargs)


@pytest.fixture
def mock_registry():
    """Fixture to mock the EMBEDDINGS_REGISTRY with MockEmbedding."""
    with patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [MockEmbedding]):
        yield


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


def test_embed_delegation(mock_registry):
    """Verify that embed() correctly delegates to the underlying model."""
    dispatcher = TextEmbedding(model_name="mock-model")
    docs = ["hello", "world"]
    list(dispatcher.embed(docs, batch_size=128, parallel=2, custom_arg="val"))

    dispatcher.model.embed_mock.assert_called_once_with(
        docs, batch_size=128, parallel=2, custom_arg="val"
    )


def test_query_embed_delegation(mock_registry):
    """Verify that query_embed() correctly delegates to the underlying model."""
    dispatcher = TextEmbedding(model_name="mock-model")
    query = "what is the meaning of life?"
    list(dispatcher.query_embed(query, custom_arg="val"))

    dispatcher.model.query_embed_mock.assert_called_once_with(query, custom_arg="val")


def test_passage_embed_delegation(mock_registry):
    """Verify that passage_embed() correctly delegates to the underlying model."""
    dispatcher = TextEmbedding(model_name="mock-model")
    passages = ["passage 1", "passage 2"]
    list(dispatcher.passage_embed(passages, custom_arg="val"))

    dispatcher.model.passage_embed_mock.assert_called_once_with(passages, custom_arg="val")


def test_token_count_delegation(mock_registry):
    """Verify that token_count() correctly delegates to the underlying model."""
    dispatcher = TextEmbedding(model_name="mock-model")
    texts = ["some text", "more text"]
    count = dispatcher.token_count(texts, batch_size=512, custom_arg="val")

    assert count == 42
    dispatcher.model.token_count_mock.assert_called_once_with(
        texts, batch_size=512, custom_arg="val"
    )


def test_embedding_size_property(mock_registry):
    """Verify that the embedding_size property returns the correct dimension."""
    dispatcher = TextEmbedding(model_name="mock-model")
    assert dispatcher.embedding_size == 4
