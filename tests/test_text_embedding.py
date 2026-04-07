from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qwen3_embed.common.model_description import ModelSource, PoolingType
from qwen3_embed.text.custom_text_embedding import CustomTextEmbedding
from qwen3_embed.text.text_embedding import TextEmbedding


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


def test_add_custom_model_and_get_embedding_size():
    """Verify adding a custom model and retrieving its embedding size."""
    model_name = "test/custom-model"
    dim = 123

    # Clear registry before test
    CustomTextEmbedding.SUPPORTED_MODELS.clear()
    CustomTextEmbedding.POSTPROCESSING_MAPPING.clear()

    TextEmbedding.add_custom_model(
        model=model_name,
        pooling=PoolingType.CLS,
        normalization=True,
        sources=ModelSource(hf="test/custom-model"),
        dim=dim,
        description="A test custom model",
        size_in_GB=0.5,
    )

    # Verify it appears in list_supported_models
    models = TextEmbedding.list_supported_models()
    assert any(m["model"] == model_name for m in models)

    # Verify get_embedding_size
    assert TextEmbedding.get_embedding_size(model_name) == dim

    # Verify it raises for unknown model
    with pytest.raises(ValueError, match="Embedding size for model .* was None"):
        TextEmbedding.get_embedding_size("non-existent-model")


def test_init_with_custom_model():
    """Verify that TextEmbedding can be initialized with a registered custom model."""
    model_name = "test/init-custom-model"

    # Clear registry and register model
    CustomTextEmbedding.SUPPORTED_MODELS.clear()
    CustomTextEmbedding.POSTPROCESSING_MAPPING.clear()

    TextEmbedding.add_custom_model(
        model=model_name,
        pooling=PoolingType.MEAN,
        normalization=False,
        sources=ModelSource(hf="test/init-custom-model"),
        dim=256,
    )

    # Mock the internal model initialization to avoid real downloads/onnx loading
    with patch(
        "qwen3_embed.text.custom_text_embedding.CustomTextEmbedding.__init__", return_value=None
    ):
        emb = TextEmbedding(model_name=model_name, lazy_load=True)
        assert isinstance(emb.model, CustomTextEmbedding)

    # Verify embedding_size property
    with patch.object(TextEmbedding, "get_embedding_size", return_value=256):
        assert emb.embedding_size == 256


def test_embed_methods_delegation():
    """Verify that embed, query_embed, passage_embed, and token_count delegate to the internal model."""
    model_name = "test/delegate-model"

    # Clear registry and register model
    CustomTextEmbedding.SUPPORTED_MODELS.clear()
    CustomTextEmbedding.POSTPROCESSING_MAPPING.clear()

    TextEmbedding.add_custom_model(
        model=model_name,
        pooling=PoolingType.CLS,
        normalization=True,
        sources=ModelSource(hf="test/delegate-model"),
        dim=256,
    )

    mock_model = MagicMock()
    mock_model.embed.return_value = iter([np.array([1, 2, 3])])
    mock_model.query_embed.return_value = iter([np.array([4, 5, 6])])
    mock_model.passage_embed.return_value = iter([np.array([7, 8, 9])])
    mock_model.token_count.return_value = 42

    with patch(
        "qwen3_embed.text.custom_text_embedding.CustomTextEmbedding.__init__", return_value=None
    ):
        emb = TextEmbedding(model_name=model_name, lazy_load=True)
        emb.model = mock_model

        # Test embed
        list(emb.embed("hello"))
        mock_model.embed.assert_called_once()

        # Test query_embed
        list(emb.query_embed("hello"))
        mock_model.query_embed.assert_called_once()

        # Test passage_embed
        list(emb.passage_embed(["hello"]))
        mock_model.passage_embed.assert_called_once()

        # Test token_count
        assert emb.token_count("hello") == 42
        mock_model.token_count.assert_called_once()
