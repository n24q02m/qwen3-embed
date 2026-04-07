from collections.abc import Iterable
from typing import Any
from unittest import mock

import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.text.text_embedding import TextEmbedding
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


class MockTextEmbedding(TextEmbeddingBase):
    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
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

    def embed(self, *args, **kwargs):
        pass

    def token_count(self, texts: str | Iterable[str], **kwargs: Any) -> int:
        return 0


@pytest.fixture
def patch_registry():
    original_registry = TextEmbedding.EMBEDDINGS_REGISTRY
    TextEmbedding.EMBEDDINGS_REGISTRY = [MockTextEmbedding] + original_registry
    yield
    TextEmbedding.EMBEDDINGS_REGISTRY = original_registry


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


def test_token_count_delegation(patch_registry):
    """Verify that token_count correctly delegates to the underlying model."""
    embedding = TextEmbedding(model_name="mock/model")

    with mock.patch.object(embedding.model, "token_count", return_value=42) as mock_token_count:
        # Test with single string
        count = embedding.token_count("hello world")
        assert count == 42
        mock_token_count.assert_called_with("hello world", batch_size=1024)

        # Test with iterable of strings
        texts = ["hello", "world"]
        count = embedding.token_count(texts)
        assert count == 42
        mock_token_count.assert_called_with(texts, batch_size=1024)

        # Test with custom batch_size and kwargs
        count = embedding.token_count(texts, batch_size=512, custom_arg="value")
        assert count == 42
        mock_token_count.assert_called_with(texts, batch_size=512, custom_arg="value")
