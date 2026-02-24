import pytest

from qwen3_embed.common.model_description import ModelSource


def test_model_source_valid_hf():
    """Test ModelSource with only hf parameter set."""
    source = ModelSource(hf="org/model")
    assert source.hf == "org/model"
    assert source.url is None


def test_model_source_valid_url():
    """Test ModelSource with only url parameter set."""
    source = ModelSource(url="https://example.com/model.tar.gz")
    assert source.url == "https://example.com/model.tar.gz"
    assert source.hf is None


def test_model_source_valid_both():
    """Test ModelSource with both hf and url parameters set."""
    source = ModelSource(hf="org/model", url="https://example.com/model.tar.gz")
    assert source.hf == "org/model"
    assert source.url == "https://example.com/model.tar.gz"


def test_model_source_invalid_none():
    """Test ModelSource raises ValueError when neither hf nor url is set."""
    with pytest.raises(ValueError, match="At least one source should be set"):
        ModelSource()
