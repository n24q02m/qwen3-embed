import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource


class TestDenseModelDescription:
    def test_dense_model_description_raises_without_dim(self) -> None:
        """
        Test that DenseModelDescription raises an AssertionError
        when initialized without a dimension (dim=None).
        """
        with pytest.raises(AssertionError, match="dim is required for dense model description"):
            DenseModelDescription(
                model="test-model",
                sources=ModelSource(hf="test/model"),
                model_file="model.onnx",
                description="Test model description",
                license="MIT",
                size_in_GB=0.5,
                dim=None,  # Explicitly setting to None to trigger the assertion
            )


class TestModelSource:
    def test_model_source_raises_without_source(self) -> None:
        """
        Test that ModelSource raises a ValueError when initialized
        without any source (hf=None and url=None).
        """
        with pytest.raises(ValueError, match="At least one source should be set"):
            ModelSource(hf=None, url=None)
