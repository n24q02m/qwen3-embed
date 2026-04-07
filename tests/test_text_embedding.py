from unittest.mock import MagicMock

import numpy as np
import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.text.text_embedding import TextEmbedding
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


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


class MockModel(TextEmbeddingBase):
    @classmethod
    def _list_supported_models(cls):
        return [
            DenseModelDescription(
                model="mock/model",
                sources=ModelSource(hf="mock/model"),
                dim=128,
                model_file="model.onnx",
                description="Mock model",
                license="MIT",
                size_in_GB=0.1,
            )
        ]

    def __init__(self, **kwargs):
        # TextEmbedding passes model_name in kwargs too
        kwargs.pop("model_name", None)
        super().__init__(model_name="mock/model", **kwargs)
        self.embed = MagicMock()  # type: ignore
        self.query_embed = MagicMock()  # type: ignore
        self.passage_embed = MagicMock()  # type: ignore
        self.token_count = MagicMock()  # type: ignore


class TestTextEmbeddingDelegation:
    def setup_method(self):
        self.original_registry = TextEmbedding.EMBEDDINGS_REGISTRY.copy()
        TextEmbedding.EMBEDDINGS_REGISTRY.insert(0, MockModel)
        self.model = TextEmbedding(model_name="mock/model")

    def teardown_method(self):
        TextEmbedding.EMBEDDINGS_REGISTRY = self.original_registry

    def test_embed_delegation(self):
        documents = ["doc1", "doc2"]
        # Mock returns an iterator
        self.model.model.embed.return_value = iter([np.array([1, 2])])  # type: ignore
        list(self.model.embed(documents, batch_size=10, parallel=2, extra="param"))
        self.model.model.embed.assert_called_once_with(documents, 10, 2, extra="param")  # type: ignore

    def test_query_embed_delegation(self):
        query = "query text"
        self.model.model.query_embed.return_value = iter([np.array([1, 2])])  # type: ignore
        list(self.model.query_embed(query, extra="param"))
        self.model.model.query_embed.assert_called_once_with(query, extra="param")  # type: ignore

    def test_passage_embed_delegation(self):
        passages = ["passage1", "passage2"]
        self.model.model.passage_embed.return_value = iter([np.array([1, 2])])  # type: ignore
        list(self.model.passage_embed(passages, extra="param"))
        self.model.model.passage_embed.assert_called_once_with(passages, extra="param")  # type: ignore

    def test_token_count_delegation(self):
        texts = ["text1", "text2"]
        self.model.model.token_count.return_value = 42  # type: ignore
        count = self.model.token_count(texts, batch_size=100, extra="param")
        assert count == 42
        self.model.model.token_count.assert_called_once_with(texts, batch_size=100, extra="param")  # type: ignore

    def test_embedding_size_delegation(self):
        # embedding_size property uses get_embedding_size class method
        assert self.model.embedding_size == 128
