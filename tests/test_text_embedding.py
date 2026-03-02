import pytest

from qwen3_embed.common.model_description import ModelSource, PoolingType
from qwen3_embed.text.text_embedding import TextEmbedding


def test_init_unsupported_model_raises_value_error():
    """Verify that TextEmbedding raises a ValueError when initialized with an unsupported model."""
    with pytest.raises(ValueError, match="is not supported in TextEmbedding"):
        TextEmbedding(model_name="unsupported-model-name")


def test_add_custom_model_duplicate_raises_error():
    """Verify that adding a custom model that already exists raises a ValueError."""
    existing_model = TextEmbedding._list_supported_models()[0].model
    with pytest.raises(ValueError, match=f"Model {existing_model} is already registered"):
        TextEmbedding.add_custom_model(
            model=existing_model,
            pooling=PoolingType.MEAN,
            normalization=False,
            sources=ModelSource(url="http://example.com/model.tar.gz"),
            dim=768,
        )
