import pytest

from qwen3_embed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder


def test_init_unsupported_model_raises_value_error():
    """Initializing TextCrossEncoder with an unsupported model should raise ValueError."""
    with pytest.raises(ValueError, match="is not supported in TextCrossEncoder"):
        TextCrossEncoder(model_name="unsupported-model-name")


def test_list_supported_models():
    """Verify that list_supported_models returns a list of dictionaries with model descriptions."""
    models = TextCrossEncoder.list_supported_models()
    assert isinstance(models, list)
    assert len(models) > 0

    # Check that each item is a dictionary with expected keys
    for model in models:
        assert isinstance(model, dict)
        assert "model" in model
        assert "description" in model
        assert "size_in_GB" in model
        assert "sources" in model
