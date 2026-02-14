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

    def test_add_custom_model_when_cls_pooling_registers_model(self):
        TextEmbedding.add_custom_model(
            model="test/cls-model",
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf="test/cls-model"),
            dim=768,
        )
        models = TextEmbedding.list_supported_models()
        assert any(m["model"] == "test/cls-model" for m in models)

    def test_add_custom_model_when_mean_pooling_registers_model(self):
        TextEmbedding.add_custom_model(
            model="test/mean-model",
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf="test/mean-model"),
            dim=512,
        )
        models = TextEmbedding.list_supported_models()
        assert any(m["model"] == "test/mean-model" for m in models)

    def test_add_custom_model_when_last_token_pooling_registers_model(self):
        TextEmbedding.add_custom_model(
            model="test/last-token-model",
            pooling=PoolingType.LAST_TOKEN,
            normalization=True,
            sources=ModelSource(hf="test/last-token-model"),
            dim=1024,
        )
        models = TextEmbedding.list_supported_models()
        assert any(m["model"] == "test/last-token-model" for m in models)

    def test_add_custom_model_when_duplicate_raises_value_error(self):
        TextEmbedding.add_custom_model(
            model="test/duplicate",
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf="test/duplicate"),
            dim=256,
        )
        with pytest.raises(ValueError, match="already registered"):
            TextEmbedding.add_custom_model(
                model="test/duplicate",
                pooling=PoolingType.CLS,
                normalization=True,
                sources=ModelSource(hf="test/duplicate"),
                dim=256,
            )
