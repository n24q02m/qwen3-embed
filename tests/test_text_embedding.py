from unittest.mock import patch

import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
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


def test_embedding_size_property():
    """Verify the embedding_size property uses the get_embedding_size class method and caches the result."""

    # We need to bypass the __init__ logic to set attributes directly
    with patch("qwen3_embed.text.text_embedding.TextEmbedding.__init__", return_value=None):
        instance = TextEmbedding()
        instance.model_name = "test-model"
        instance._embedding_size = None

        # Call the property, mocking the class method
        with patch.object(TextEmbedding, "get_embedding_size", return_value=768) as mock_get_size:
            # First access should call get_embedding_size
            assert instance.embedding_size == 768
            mock_get_size.assert_called_once_with("test-model")

            # Second access should return the cached value without calling get_embedding_size again
            assert instance.embedding_size == 768
            assert mock_get_size.call_count == 1


def test_get_embedding_size_class_method():
    """Verify the get_embedding_size class method returns the correct dimension and handles case-insensitivity."""
    mock_description = DenseModelDescription(
        model="test-model",
        sources=ModelSource(hf="test-model"),
        dim=512,
        model_file="model.onnx",
        description="test",
        license="MIT",
        size_in_GB=0.1,
    )

    with patch.object(TextEmbedding, "_list_supported_models", return_value=[mock_description]):
        # Case-insensitive check
        assert TextEmbedding.get_embedding_size("test-model") == 512
        assert TextEmbedding.get_embedding_size("TEST-MODEL") == 512

        # Error handling for unknown model
        with pytest.raises(ValueError, match="Embedding size for model unknown-model was None"):
            TextEmbedding.get_embedding_size("unknown-model")
