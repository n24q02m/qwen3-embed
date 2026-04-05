from typing import Any
from unittest.mock import MagicMock

import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.text.text_embedding import TextEmbedding
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


class MockModel(TextEmbeddingBase):
    embed: MagicMock
    query_embed: MagicMock
    passage_embed: MagicMock
    token_count: MagicMock

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return [
            DenseModelDescription(
                model="mock/model",
                dim=128,
                sources=ModelSource(hf="mock/model"),
                model_file="model.onnx",
                description="Mock model",
                license="MIT",
                size_in_GB=0.1,
            )
        ]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.embed = MagicMock(return_value=iter([]))
        self.query_embed = MagicMock(return_value=iter([]))
        self.passage_embed = MagicMock(return_value=iter([]))
        self.token_count = MagicMock(return_value=42)


@pytest.fixture
def registered_mock_model():
    original_registry = TextEmbedding.EMBEDDINGS_REGISTRY.copy()
    TextEmbedding.EMBEDDINGS_REGISTRY.insert(0, MockModel)
    yield
    TextEmbedding.EMBEDDINGS_REGISTRY = original_registry


def test_query_embed_delegation(registered_mock_model):
    """Verify that TextEmbedding.query_embed delegates to the underlying model."""
    model = TextEmbedding(model_name="mock/model")
    assert isinstance(model.model, MockModel)

    query = "test query"
    kwargs = {"some_arg": "value"}

    # query_embed is a generator in TextEmbedding, so we must consume it
    list(model.query_embed(query, **kwargs))

    model.model.query_embed.assert_called_once_with(query, **kwargs)


def test_embed_delegation(registered_mock_model):
    """Verify that TextEmbedding.embed delegates to the underlying model."""
    model = TextEmbedding(model_name="mock/model")
    assert isinstance(model.model, MockModel)

    docs = ["doc1", "doc2"]
    batch_size = 10
    parallel = 2
    kwargs = {"extra": "stuff"}

    list(model.embed(docs, batch_size=batch_size, parallel=parallel, **kwargs))

    model.model.embed.assert_called_once_with(docs, batch_size, parallel, **kwargs)


def test_passage_embed_delegation(registered_mock_model):
    """Verify that TextEmbedding.passage_embed delegates to the underlying model."""
    model = TextEmbedding(model_name="mock/model")
    assert isinstance(model.model, MockModel)

    texts = ["text1", "text2"]
    kwargs = {"foo": "bar"}

    list(model.passage_embed(texts, **kwargs))

    model.model.passage_embed.assert_called_once_with(texts, **kwargs)


def test_token_count_delegation(registered_mock_model):
    """Verify that TextEmbedding.token_count delegates to the underlying model."""
    model = TextEmbedding(model_name="mock/model")
    assert isinstance(model.model, MockModel)

    texts = "some text"
    batch_size = 123
    kwargs = {"baz": "qux"}

    count = model.token_count(texts, batch_size=batch_size, **kwargs)

    assert count == 42
    model.model.token_count.assert_called_once_with(texts, batch_size=batch_size, **kwargs)
