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


def test_embedding_size_delegation():
    """Verify that embedding_size property delegates to the underlying model."""
    with patch("qwen3_embed.text.text_embedding.TextEmbedding.__init__", return_value=None):
        te = TextEmbedding()
        te.model = MagicMock()
        te.model.embedding_size = 512
        te._embedding_size = None

        assert te.embedding_size == 512
        te.model.embedding_size = 1024
        # Verify caching: it should still be 512 because it was cached
        assert te.embedding_size == 512


def test_get_embedding_size():
    """Verify that get_embedding_size class method returns correct size."""
    # Use a known model from the registry
    models = TextEmbedding.list_supported_models()
    if models:
        model_name = models[0]["model"]
        expected_dim = models[0]["dim"]
        assert TextEmbedding.get_embedding_size(model_name) == expected_dim

    with pytest.raises(ValueError, match="Embedding size for model"):
        TextEmbedding.get_embedding_size("non-existent-model")
