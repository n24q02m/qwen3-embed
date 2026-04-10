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


def test_add_custom_model():
    """Verify that add_custom_model correctly adds a new model to the supported models."""
    # Clear custom model registry to ensure a clean state
    CustomTextEmbedding.SUPPORTED_MODELS.clear()
    CustomTextEmbedding.POSTPROCESSING_MAPPING.clear()

    model_name = "test/custom-model"
    dim = 128
    description = "A test custom model"

    TextEmbedding.add_custom_model(
        model=model_name,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf="test/custom-model"),
        dim=dim,
        description=description,
    )

    models = TextEmbedding.list_supported_models()
    custom_model = next((m for m in models if m["model"] == model_name), None)

    assert custom_model is not None
    assert custom_model["dim"] == dim
    assert custom_model["description"] == description

    # Clean up after test
    CustomTextEmbedding.SUPPORTED_MODELS.clear()
    CustomTextEmbedding.POSTPROCESSING_MAPPING.clear()
