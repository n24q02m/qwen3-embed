import pytest
from qwen3_embed.common.model_description import ModelSource, DenseModelDescription

def test_model_source_validation_no_source() -> None:
    """ModelSource should raise ValueError if both hf and url are None."""
    with pytest.raises(ValueError, match="At least one source should be set"):
        ModelSource(hf=None, url=None)

def test_model_source_valid_hf() -> None:
    """ModelSource should initialize correctly with hf source."""
    source = ModelSource(hf="org/model")
    assert source.hf == "org/model"
    assert source.url is None

def test_model_source_valid_url() -> None:
    """ModelSource should initialize correctly with url source."""
    source = ModelSource(url="https://example.com/model.tar.gz")
    assert source.url == "https://example.com/model.tar.gz"
    assert source.hf is None

def test_dense_model_description_validation_no_dim() -> None:
    """DenseModelDescription should raise AssertionError if dim is None."""
    source = ModelSource(hf="test/test")

    with pytest.raises(AssertionError, match="dim is required"):
        DenseModelDescription(
            model="test-model",
            sources=source,
            model_file="model.onnx",
            description="test description",
            license="MIT",
            size_in_GB=0.1,
            dim=None
        )
