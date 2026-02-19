import pytest

from qwen3_embed.text.text_embedding import TextEmbedding


class TestTextEmbedding:
    """Tests for TextEmbedding class methods."""

    def test_get_embedding_size_unsupported_model(self):
        """Test that get_embedding_size raises ValueError for unsupported models."""
        with pytest.raises(ValueError, match="Embedding size for model .* was None"):
            TextEmbedding.get_embedding_size("non-existent-model")

    def test_get_embedding_size_supported_model(self):
        """Test that get_embedding_size returns correct size for a supported model."""
        # Dynamically get a supported model to avoid hardcoding
        supported_models = TextEmbedding.list_supported_models()
        assert len(supported_models) > 0, "No supported models found to test against"

        model_info = supported_models[0]
        model_name = model_info["model"]
        expected_size = model_info["dim"]

        size = TextEmbedding.get_embedding_size(model_name)
        assert size == expected_size
