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


def test_get_embedding_size_valid_model():
    """Verify that get_embedding_size returns the correct dimension for a valid model."""
    model_name = "n24q02m/Qwen3-Embedding-0.6B-ONNX"
    size = TextEmbedding.get_embedding_size(model_name)
    assert isinstance(size, int)
    assert size == 1024


def test_get_embedding_size_case_insensitive():
    """Verify that get_embedding_size is case-insensitive."""
    model_name = "N24Q02M/QWEN3-EMBEDDING-0.6B-ONNX"
    size = TextEmbedding.get_embedding_size(model_name)
    assert size == 1024


def test_get_embedding_size_invalid_model_raises_value_error():
    """Verify that get_embedding_size raises a ValueError for an invalid model name."""
    invalid_model = "non-existent-model"
    with pytest.raises(ValueError, match=f"Embedding size for model {invalid_model} was None"):
        TextEmbedding.get_embedding_size(invalid_model)
