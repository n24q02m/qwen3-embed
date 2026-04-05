from unittest.mock import MagicMock, patch

import pytest

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
    """Verify that the embedding_size property correctly delegates to the underlying model."""
    # Mocking the dispatcher to avoid actual model initialization
    with patch("qwen3_embed.text.text_embedding.TextEmbedding.__init__", return_value=None):
        embedding = TextEmbedding()
        embedding.model = MagicMock()
        embedding.model.embedding_size = 768
        embedding._embedding_size = None

        assert embedding.embedding_size == 768
        # Test caching
        embedding.model.embedding_size = 1024
        assert embedding.embedding_size == 768


def test_embedding_size_integration():
    """Verify embedding_size on a real (but lazy-loaded) TextEmbedding instance."""
    model_name = "n24q02m/Qwen3-Embedding-0.6B-ONNX"
    with patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding.__init__", return_value=None):
        embedding = TextEmbedding(model_name=model_name, lazy_load=True)
        # We need to mock the model_description.dim since we mocked __init__
        embedding.model.model_description = MagicMock()
        embedding.model.model_description.dim = 1024

        assert embedding.embedding_size == 1024


def test_get_embedding_size_class_method():
    """Verify the get_embedding_size class method."""
    model_name = "n24q02m/Qwen3-Embedding-0.6B-ONNX"
    size = TextEmbedding.get_embedding_size(model_name)
    assert size == 1024


def test_get_embedding_size_invalid_model():
    """Verify that get_embedding_size raises ValueError for unknown models."""
    with pytest.raises(ValueError, match="Embedding size for model .* was None"):
        TextEmbedding.get_embedding_size("invalid-model")
