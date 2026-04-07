import pytest
from unittest.mock import patch, MagicMock
from collections.abc import Iterable
from typing import Any

from qwen3_embed.text.text_embedding import TextEmbedding
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase
from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.common.types import NumpyArray


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
                sources=ModelSource(hf="mock/model"),
                dim=123,
                model_file="model.onnx",
                description="Mock model for testing",
                license="MIT",
                size_in_GB=0.1,
            )
        ]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._embedding_size = 123

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        return []

    def token_count(
        self, texts: str | Iterable[str], batch_size: int = 1024, **kwargs: Any
    ) -> int:
        return 0


def test_get_embedding_size_supported():
    """Verify that get_embedding_size returns the correct dimension for a supported model."""
    with patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [MockEmbeddingModel]):
        size = TextEmbedding.get_embedding_size("mock-model")
        assert size == 123


def test_get_embedding_size_unsupported():
    """Verify that get_embedding_size raises ValueError for an unsupported model."""
    with (
        patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [MockEmbeddingModel]),
        pytest.raises(ValueError, match="Embedding size for model unknown-model was None"),
    ):
        TextEmbedding.get_embedding_size("unknown-model")


def test_embedding_size_property():
    """Verify that the embedding_size property returns the correct value."""
    mock_instance = MagicMock(spec=MockEmbeddingModel)
    # Ensure the mock instance has the expected behavior if needed,
    # but the property calls TextEmbedding.get_embedding_size

    with (
        patch.object(
            TextEmbedding,
            "EMBEDDINGS_REGISTRY",
            [
                MagicMock(
                    return_value=mock_instance,
                    _list_supported_models=MockEmbeddingModel._list_supported_models,
                )
            ],
        ),
    ):
        embedding = TextEmbedding(model_name="mock-model")
        embedding.model = mock_instance
        embedding.model_name = "mock-model"

        # The property calls get_embedding_size(self.model_name)
        assert embedding.embedding_size == 123
        # Check if it's cached
        assert embedding._embedding_size == 123
