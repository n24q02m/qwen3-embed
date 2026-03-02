import pytest

from qwen3_embed.text.text_embedding import TextEmbedding


def test_init_unsupported_model_raises_value_error():
    """Verify that TextEmbedding raises a ValueError when initialized with an unsupported model."""
    with pytest.raises(
        ValueError,
        match=r"^Model unsupported-model-name is not supported in TextEmbedding\. "
        r"Please check the supported models using `TextEmbedding\.list_supported_models\(\)`$",
    ):
        TextEmbedding(model_name="unsupported-model-name")
