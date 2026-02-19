import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource


def test_create_DenseModelDescription_with_missing_dim_raises_AssertionError():
    """
    Test that DenseModelDescription raises an AssertionError if `dim` is not provided.
    """
    with pytest.raises(AssertionError, match="dim is required for dense model description"):
        DenseModelDescription(
            model="test-model",
            sources=ModelSource(hf="test/model"),
            model_file="model.onnx",
            description="Test Description",
            license="MIT",
            size_in_GB=0.1,
            dim=None,
        )


def test_create_ModelSource_with_missing_sources_raises_ValueError():
    """
    Test that ModelSource raises a ValueError if both `hf` and `url` are None.
    """
    with pytest.raises(ValueError, match="At least one source should be set"):
        ModelSource(hf=None, url=None)
