"""Tests for custom model registration via TextCrossEncoder.add_custom_model."""

import pytest

from qwen3_embed.common.model_description import BaseModelDescription, ModelSource
from qwen3_embed.rerank.cross_encoder.custom_text_cross_encoder import CustomTextCrossEncoder
from qwen3_embed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder


class TestCustomCrossEncoderRegistration:
    """Verify custom model registration for CrossEncoder works."""

    def setup_method(self):
        """Clear custom model registry between tests."""
        CustomTextCrossEncoder.SUPPORTED_MODELS.clear()

    def test_register_model(self):
        """Test adding a basic model."""
        TextCrossEncoder.add_custom_model(
            model_description=BaseModelDescription(
                model="test/model",
                sources=ModelSource(hf="test/model"),
                model_file="onnx/model.onnx",
                description="",
                license="",
                size_in_GB=0.0,
                additional_files=[],
            ),
        )
        models = TextCrossEncoder.list_supported_models()
        assert any(m["model"] == "test/model" for m in models)

    def test_duplicate_model_raises_case_insensitive(self):
        """Test adding a duplicate model with different casing raises ValueError."""
        TextCrossEncoder.add_custom_model(
            model_description=BaseModelDescription(
                model="Test/Duplicate",
                sources=ModelSource(hf="Test/Duplicate"),
                model_file="onnx/model.onnx",
                description="",
                license="",
                size_in_GB=0.0,
                additional_files=[],
            ),
        )

        # Verify it was added
        models = TextCrossEncoder.list_supported_models()
        assert any(m["model"] == "Test/Duplicate" for m in models)

        # Try adding again with different case
        with pytest.raises(ValueError, match="already registered"):
            TextCrossEncoder.add_custom_model(
                model_description=BaseModelDescription(
                    model="test/duplicate",
                    sources=ModelSource(hf="test/duplicate"),
                    model_file="onnx/model.onnx",
                    description="",
                    license="",
                    size_in_GB=0.0,
                    additional_files=[],
                ),
            )
