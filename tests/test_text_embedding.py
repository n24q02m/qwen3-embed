from unittest.mock import MagicMock, patch

import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.text.text_embedding import TextEmbedding


def test_init_unsupported_model_raises_value_error():
    """Verify that TextEmbedding raises a ValueError when initialized with an unsupported model."""
    with pytest.raises(ValueError, match="is not supported in TextEmbedding"):
        TextEmbedding(model_name="unsupported-model-name")


def test_list_supported_models():
    """Verify that list_supported_models returns a list of dictionaries with model descriptions."""
    models = TextEmbedding.list_supported_models()
    assert isinstance(models, list)
    assert len(models) > 0

    # Check that each item is a dictionary with expected keys
    for model in models:
        assert isinstance(model, dict)
        assert "model" in model
        assert "dim" in model
        assert "description" in model
        assert "size_in_GB" in model
        assert "sources" in model


def test_token_count_delegation():
    """Verify that TextEmbedding.token_count delegates correctly to the underlying model."""
    model_name = "test-model"
    mock_model_class = MagicMock()
    mock_model_instance = mock_model_class.return_value
    mock_model_instance.token_count.return_value = 123

    # Mock _list_supported_models to return our test model
    mock_model_class._list_supported_models.return_value = [
        DenseModelDescription(
            model=model_name,
            sources=ModelSource(hf=model_name),
            dim=128,
            model_file="model.onnx",
            description="Test",
            license="MIT",
            size_in_GB=0.1,
        )
    ]

    with patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [mock_model_class]):
        te = TextEmbedding(model_name=model_name)

        # Test single string input
        assert te.token_count("test text") == 123
        mock_model_instance.token_count.assert_called_once_with("test text", batch_size=1024)

        mock_model_instance.token_count.reset_mock()

        # Test iterable input with custom batch_size and kwargs
        docs = ["a", "b", "c"]
        assert te.token_count(docs, batch_size=64, some_param="value") == 123
        mock_model_instance.token_count.assert_called_once_with(
            docs, batch_size=64, some_param="value"
        )
