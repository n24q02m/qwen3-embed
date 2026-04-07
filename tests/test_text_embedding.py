from collections.abc import Iterable
from typing import Any
from unittest.mock import MagicMock, patch

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


class MockEmbeddingModel(TextEmbeddingBase):
    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return [
            DenseModelDescription(
                model="mock-model",
                sources=ModelSource(hf="mock-hf"),
                dim=128,
                model_file="model.onnx",
                description="mock description",
                license="mock license",
                size_in_GB=0.1,
            )
        ]

    def __init__(self, *args: Any, **kwargs: Any):
        # The dispatcher passes model_name in kwargs.
        model_name = kwargs.pop("model_name", "mock-model")
        super().__init__(model_name=model_name)
        self.token_count = MagicMock(return_value=42)

    def embed(self, *args: Any, **kwargs: Any) -> Any:
        pass


def test_token_count_delegation():
    """Verify that TextEmbedding.token_count delegates to the underlying model."""
    with patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [MockEmbeddingModel]):
        embedding = TextEmbedding(model_name="mock-model")

        # Test with single string
        count = embedding.token_count("hello")
        assert count == 42
        embedding.model.token_count.assert_called_with("hello", batch_size=1024)

        # Test with iterable and custom batch_size/kwargs
        embedding.model.token_count.reset_mock()
        embedding.model.token_count.return_value = 100
        count = embedding.token_count(["hello", "world"], batch_size=512, custom_arg="value")
        assert count == 100
        embedding.model.token_count.assert_called_with(
            ["hello", "world"], batch_size=512, custom_arg="value"
        )
