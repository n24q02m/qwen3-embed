from unittest.mock import MagicMock, patch

import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.text.text_embedding import TextEmbedding


@pytest.fixture
def mock_registry():
    mock_model_desc = DenseModelDescription(
        model="mock-model",
        dim=768,
        sources=ModelSource(hf="mock/model"),
        model_file="onnx/model.onnx",
        description="",
        license="",
        size_in_GB=0.0,
    )

    class MockEmbedding:
        def __init__(self, *args, **kwargs):
            self.embed = MagicMock(return_value=iter([]))
            self.query_embed = MagicMock(return_value=iter([]))
            self.passage_embed = MagicMock(return_value=iter([]))
            self.token_count = MagicMock(return_value=10)
            self.model_name = "mock-model"

        @classmethod
        def _list_supported_models(cls):
            return [mock_model_desc]

    with patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [MockEmbedding]):
        # Also need to patch the registry on the class level if needed,
        # but patch.object should handle it for the duration of the test.
        yield MockEmbedding


def test_init_unsupported_model_raises_value_error():
    """Verify that TextEmbedding raises a ValueError when initialized with an unsupported model."""
    with pytest.raises(ValueError, match="is not supported in TextEmbedding"):
        TextEmbedding(model_name="unsupported-model-name")


def test_list_supported_models(mock_registry):
    """Verify that list_supported_models returns a list of dictionaries with model descriptions."""
    models = TextEmbedding.list_supported_models()
    assert isinstance(models, list)
    assert len(models) == 1
    assert models[0]["model"] == "mock-model"


def test_delegation_methods(mock_registry):
    """Verify that TextEmbedding correctly delegates calls to the underlying model."""
    model = TextEmbedding(model_name="mock-model")
    inner_model = model.model

    # Test embed
    list(model.embed(["test doc"], batch_size=32, parallel=2, extra="arg"))
    inner_model.embed.assert_called_once_with(["test doc"], 32, 2, extra="arg")

    # Test query_embed
    list(model.query_embed("test query", task="search"))
    inner_model.query_embed.assert_called_once_with("test query", task="search")

    # Test passage_embed
    list(model.passage_embed(["test passage"], dim=128))
    inner_model.passage_embed.assert_called_once_with(["test passage"], dim=128)

    # Test token_count
    count = model.token_count(["test text"], batch_size=128)
    assert count == 10
    inner_model.token_count.assert_called_once_with(["test text"], batch_size=128)


def test_embedding_size(mock_registry):
    """Verify that embedding_size returns the correct dimension from the registry."""
    # Test instance property
    model = TextEmbedding(model_name="mock-model")
    assert model.embedding_size == 768

    # Test class method
    assert TextEmbedding.get_embedding_size("mock-model") == 768


def test_get_embedding_size_raises_value_error(mock_registry):
    """Verify that get_embedding_size raises ValueError for unknown models."""
    with pytest.raises(ValueError, match="Embedding size for model unknown-model was None"):
        TextEmbedding.get_embedding_size("unknown-model")


def test_add_custom_model(mock_registry):
    """Verify that add_custom_model calls CustomTextEmbedding.add_model."""
    from qwen3_embed.common.model_description import PoolingType
    from qwen3_embed.text.custom_text_embedding import CustomTextEmbedding

    with patch.object(CustomTextEmbedding, "add_model") as mock_add:
        TextEmbedding.add_custom_model(
            model="new-model",
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf="new/model"),
            dim=384,
        )
        mock_add.assert_called_once()
        args, kwargs = mock_add.call_args
        assert args[0].model == "new-model"
        assert args[0].dim == 384
        assert kwargs["pooling"] == PoolingType.MEAN
        assert kwargs["normalization"] is True


def test_add_custom_model_already_registered(mock_registry):
    """Verify that add_custom_model raises ValueError if model name is already taken."""
    from qwen3_embed.common.model_description import PoolingType

    with pytest.raises(ValueError, match="is already registered in TextEmbedding"):
        TextEmbedding.add_custom_model(
            model="mock-model",
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf="new/model"),
            dim=384,
        )
