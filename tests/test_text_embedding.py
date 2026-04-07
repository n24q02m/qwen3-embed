from collections.abc import Iterable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource, PoolingType
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.text.custom_text_embedding import CustomTextEmbedding
from qwen3_embed.text.text_embedding import TextEmbedding
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


class MockModel(TextEmbeddingBase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__("mock-model", None, None)
        self.embed = MagicMock(return_value=iter([]))
        self.query_embed = MagicMock(return_value=iter([]))
        self.passage_embed = MagicMock(return_value=iter([]))
        self.token_count = MagicMock(return_value=42)

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return [
            DenseModelDescription(
                model="mock-model",
                sources=ModelSource(hf="mock/model"),
                dim=128,
                model_file="model.onnx",
                description="Mock model",
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
    ) -> Iterable[NumpyArray]:
        return self.embed(documents, batch_size, parallel, **kwargs)

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        return self.query_embed(query, **kwargs)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        return self.passage_embed(texts, **kwargs)

    def token_count(
        self, texts: str | Iterable[str], batch_size: int = 1024, **kwargs: Any
    ) -> int:
        return self.token_count(texts, batch_size=batch_size, **kwargs)


@pytest.fixture
def mock_registry():
    original_registry = TextEmbedding.EMBEDDINGS_REGISTRY
    TextEmbedding.EMBEDDINGS_REGISTRY = [MockModel]
    yield
    TextEmbedding.EMBEDDINGS_REGISTRY = original_registry


def test_init_success(mock_registry):
    """Verify that TextEmbedding initializes correctly with a supported model."""
    emb = TextEmbedding(model_name="mock-model")
    assert isinstance(emb.model, MockModel)
    assert emb.model_name == "mock-model"


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


def test_embedding_size_property(mock_registry):
    """Verify that the embedding_size property returns the correct dimension and uses caching."""
    emb = TextEmbedding(model_name="mock-model")
    assert emb.embedding_size == 128
    # Test caching: calling it again should not trigger get_embedding_size
    with patch.object(
        TextEmbedding, "get_embedding_size", side_effect=TextEmbedding.get_embedding_size
    ) as mock_get:
        _ = emb.embedding_size
        assert mock_get.call_count == 0


def test_get_embedding_size_case_insensitive(mock_registry):
    """Verify that get_embedding_size is case-insensitive."""
    assert TextEmbedding.get_embedding_size("MOCK-MODEL") == 128


def test_get_embedding_size_not_found():
    """Verify that get_embedding_size raises a ValueError for an unknown model."""
    with pytest.raises(ValueError, match="Embedding size for model unknown was None"):
        TextEmbedding.get_embedding_size("unknown")


def test_delegation_embed(mock_registry):
    """Verify that embed calls are delegated to the underlying model."""
    emb = TextEmbedding(model_name="mock-model")
    list(emb.embed(["hi"], batch_size=10))
    emb.model.embed.assert_called_once_with(["hi"], 10, None)


def test_delegation_query_embed(mock_registry):
    """Verify that query_embed calls are delegated to the underlying model."""
    emb = TextEmbedding(model_name="mock-model")
    list(emb.query_embed("query"))
    emb.model.query_embed.assert_called_once_with("query")


def test_delegation_passage_embed(mock_registry):
    """Verify that passage_embed calls are delegated to the underlying model."""
    emb = TextEmbedding(model_name="mock-model")
    list(emb.passage_embed(["text"]))
    emb.model.passage_embed.assert_called_once_with(["text"])


def test_delegation_token_count(mock_registry):
    """Verify that token_count calls are delegated to the underlying model."""
    emb = TextEmbedding(model_name="mock-model")
    count = emb.token_count(["text"], batch_size=100)
    assert count == 42
    emb.model.token_count.assert_called_once_with(["text"], batch_size=100)


def test_add_custom_model_comprehensive():
    """Verify that add_custom_model correctly registers a new model."""
    # Reset CustomTextEmbedding registry
    CustomTextEmbedding.SUPPORTED_MODELS.clear()

    TextEmbedding.add_custom_model(
        model="new-custom-model",
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf="new/custom"),
        dim=256,
        description="A new custom model",
        license="Apache-2.0",
        size_in_gb=0.5,
        additional_files=["config.json"],
    )

    models = TextEmbedding.list_supported_models()
    found = False
    for m in models:
        if m["model"] == "new-custom-model":
            assert m["dim"] == 256
            assert m["description"] == "A new custom model"
            assert m["license"] == "Apache-2.0"
            assert m["size_in_GB"] == 0.5
            assert m["sources"]["hf"] == "new/custom"
            assert "config.json" in m["additional_files"]
            found = True
            break
    assert found


def test_add_custom_model_duplicate():
    """Verify that add_custom_model raises a ValueError for duplicate model names."""
    model_name = "duplicate-check"
    # Ensure it's not already there
    CustomTextEmbedding.SUPPORTED_MODELS.clear()

    TextEmbedding.add_custom_model(
        model=model_name,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf="test/dup"),
        dim=128,
    )

    with pytest.raises(ValueError, match="is already registered"):
        TextEmbedding.add_custom_model(
            model=model_name,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf="test/dup"),
            dim=128,
        )
