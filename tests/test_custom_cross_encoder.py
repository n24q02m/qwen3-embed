"""Tests for custom model registration via TextCrossEncoder.add_custom_model."""

import pytest

from qwen3_embed.common.model_description import ModelSource
from qwen3_embed.rerank.cross_encoder.custom_text_cross_encoder import CustomTextCrossEncoder
from qwen3_embed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder


class TestCustomCrossEncoderRegistration:
    """Verify custom model registration for CrossEncoder."""

    def setup_method(self):
        """Clear custom model registry between tests."""
        CustomTextCrossEncoder.SUPPORTED_MODELS.clear()

    def test_register_custom_model(self):
        """Test successful registration of a custom model."""
        model_name = "test/custom-cross-encoder"
        TextCrossEncoder.add_custom_model(
            model=model_name,
            sources=ModelSource(hf=model_name),
            model_file="onnx/model.onnx",
        )
        models = TextCrossEncoder.list_supported_models()
        assert any(m["model"] == model_name for m in models)

    def test_duplicate_model_raises(self):
        """Test that registering a duplicate model raises ValueError."""
        model_name = "test/duplicate-cross-encoder"
        TextCrossEncoder.add_custom_model(
            model=model_name,
            sources=ModelSource(hf=model_name),
            model_file="onnx/model.onnx",
        )

        with pytest.raises(ValueError, match="already registered"):
            TextCrossEncoder.add_custom_model(
                model=model_name,
                sources=ModelSource(hf=model_name),
                model_file="onnx/model.onnx",
            )
