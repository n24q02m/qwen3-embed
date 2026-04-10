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

def test_get_embedding_size():
    """Verify that get_embedding_size returns the correct dimension for supported models."""
    # Get a supported model from the list
    supported_models = TextEmbedding.list_supported_models()
    assert len(supported_models) > 0
    model_info = supported_models[0]
    model_name = model_info["model"]
    expected_dim = model_info["dim"]

    # Test exact match
    assert TextEmbedding.get_embedding_size(model_name) == expected_dim

    # Test case-insensitivity
    assert TextEmbedding.get_embedding_size(model_name.lower()) == expected_dim
    assert TextEmbedding.get_embedding_size(model_name.upper()) == expected_dim

    # Test unknown model
    with pytest.raises(ValueError, match="Embedding size for model unknown-model was None"):
        TextEmbedding.get_embedding_size("unknown-model")
