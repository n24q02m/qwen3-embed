import pytest

from qwen3_embed.text.text_embedding import TextEmbedding


def test_init_unsupported_model_raises_value_error():
    """Verify that TextEmbedding raises a ValueError when initialized with an unsupported model."""
    with pytest.raises(ValueError, match="is not supported in TextEmbedding"):
        TextEmbedding(model_name="unsupported-model-name")

def test_get_embedding_size_unknown_model_raises_value_error():
    """Verify that TextEmbedding.get_embedding_size raises a ValueError when called with an unknown model."""
    with pytest.raises(ValueError, match="Embedding size for model unknown-model was None. Available model names:"):
        TextEmbedding.get_embedding_size("unknown-model")
