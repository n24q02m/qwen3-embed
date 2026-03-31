"""Tests for custom model registration via TextEmbedding.add_custom_model."""

import pytest

from qwen3_embed.common.model_description import ModelSource, PoolingType
from qwen3_embed.text.custom_text_embedding import CustomTextEmbedding
from qwen3_embed.text.text_embedding import TextEmbedding


class TestCustomModelRegistration:
    """Verify custom model registration works for all pooling types."""

    def setup_method(self):
        """Clear custom model registry between tests."""
        CustomTextEmbedding.SUPPORTED_MODELS.clear()
        CustomTextEmbedding.POSTPROCESSING_MAPPING.clear()

    def test_register_cls_pooling_model(self):
        from qwen3_embed.common.model_description import DenseModelDescription

        TextEmbedding.add_custom_model(
            DenseModelDescription(
                model="test/cls-model",
                sources=ModelSource(hf="test/cls-model"),
                dim=768,
                model_file="onnx/model.onnx",
                description="",
                license="",
                size_in_GB=0.0,
            ),
            pooling=PoolingType.CLS,
            normalization=True,
        )
        models = TextEmbedding.list_supported_models()
        assert any(m["model"] == "test/cls-model" for m in models)

    def test_register_mean_pooling_model(self):
        from qwen3_embed.common.model_description import DenseModelDescription

        TextEmbedding.add_custom_model(
            DenseModelDescription(
                model="test/mean-model",
                sources=ModelSource(hf="test/mean-model"),
                dim=512,
                model_file="onnx/model.onnx",
                description="",
                license="",
                size_in_GB=0.0,
            ),
            pooling=PoolingType.MEAN,
            normalization=True,
        )
        models = TextEmbedding.list_supported_models()
        assert any(m["model"] == "test/mean-model" for m in models)

    def test_register_last_token_pooling_model(self):
        from qwen3_embed.common.model_description import DenseModelDescription

        TextEmbedding.add_custom_model(
            DenseModelDescription(
                model="test/last-token-model",
                sources=ModelSource(hf="test/last-token-model"),
                dim=1024,
                model_file="onnx/model.onnx",
                description="",
                license="",
                size_in_GB=0.0,
            ),
            pooling=PoolingType.LAST_TOKEN,
            normalization=True,
        )
        models = TextEmbedding.list_supported_models()
        assert any(m["model"] == "test/last-token-model" for m in models)

    def test_duplicate_model_raises(self):
        from qwen3_embed.common.model_description import DenseModelDescription

        TextEmbedding.add_custom_model(
            DenseModelDescription(
                model="test/duplicate",
                sources=ModelSource(hf="test/duplicate"),
                dim=256,
                model_file="onnx/model.onnx",
                description="",
                license="",
                size_in_GB=0.0,
            ),
            pooling=PoolingType.CLS,
            normalization=True,
        )
        with pytest.raises(ValueError, match="already registered"):
            TextEmbedding.add_custom_model(
                DenseModelDescription(
                    model="test/duplicate",
                    sources=ModelSource(hf="test/duplicate"),
                    dim=256,
                    model_file="onnx/model.onnx",
                    description="",
                    license="",
                    size_in_GB=0.0,
                ),
                pooling=PoolingType.CLS,
                normalization=True,
            )
