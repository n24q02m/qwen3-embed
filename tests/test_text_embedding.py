from unittest.mock import patch

import pytest

from qwen3_embed.text.text_embedding import TextEmbedding
from tests.mock_embedding_model import MockEmbeddingModel


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


class TestTextEmbeddingDelegation:
    @pytest.fixture
    def mock_text_embedding(self):
        with patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [MockEmbeddingModel]):
            yield TextEmbedding(model_name="mock-model")

    def test_embed_delegation_str(self, mock_text_embedding):
        list(mock_text_embedding.embed("test doc", batch_size=10, parallel=2, custom_kwarg="val"))
        mock_model = mock_text_embedding.model
        assert mock_model.calls[0][0] == "embed"
        assert mock_model.calls[0][1] == "test doc"
        assert mock_model.calls[0][2] == 10
        assert mock_model.calls[0][3] == 2
        assert mock_model.calls[0][4] == {"custom_kwarg": "val"}

    def test_embed_delegation_iterable(self, mock_text_embedding):
        docs = ["doc 1", "doc 2"]
        list(mock_text_embedding.embed(docs, batch_size=10, parallel=2, custom_kwarg="val"))
        mock_model = mock_text_embedding.model
        assert mock_model.calls[0][0] == "embed"
        # docs are wrapped in iter_checked_texts
        assert list(mock_model.calls[0][1]) == docs
        assert mock_model.calls[0][2] == 10
        assert mock_model.calls[0][3] == 2
        assert mock_model.calls[0][4] == {"custom_kwarg": "val"}

    def test_query_embed_delegation_str(self, mock_text_embedding):
        list(mock_text_embedding.query_embed("test query", custom_kwarg="val"))
        mock_model = mock_text_embedding.model
        assert mock_model.calls[0][0] == "query_embed"
        assert mock_model.calls[0][1] == "test query"
        assert mock_model.calls[0][2] == {"custom_kwarg": "val"}

    def test_query_embed_delegation_iterable(self, mock_text_embedding):
        queries = ["query 1", "query 2"]
        list(mock_text_embedding.query_embed(queries, custom_kwarg="val"))
        mock_model = mock_text_embedding.model
        assert mock_model.calls[0][0] == "query_embed"
        # queries are wrapped in iter_checked_texts
        assert list(mock_model.calls[0][1]) == queries
        assert mock_model.calls[0][2] == {"custom_kwarg": "val"}

    def test_passage_embed_delegation(self, mock_text_embedding):
        texts = ["passage 1", "passage 2"]
        list(mock_text_embedding.passage_embed(texts, custom_kwarg="val"))
        mock_model = mock_text_embedding.model
        assert mock_model.calls[0][0] == "passage_embed"
        # texts are wrapped in iter_checked_texts
        assert list(mock_model.calls[0][1]) == texts
        assert mock_model.calls[0][2] == {"custom_kwarg": "val"}

    def test_token_count_delegation(self, mock_text_embedding):
        count = mock_text_embedding.token_count("test text", batch_size=5, custom_kwarg="val")
        mock_model = mock_text_embedding.model
        assert count == 42
        assert mock_model.calls[0][0] == "token_count"
        assert mock_model.calls[0][1] == "test text"
        assert mock_model.calls[0][2] == {"batch_size": 5, "custom_kwarg": "val"}

    def test_embedding_size(self, mock_text_embedding):
        assert mock_text_embedding.embedding_size == 128

    def test_get_embedding_size_raises(self, mock_text_embedding):
        with (
            patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [MockEmbeddingModel]),
            pytest.raises(ValueError, match="Embedding size for model unknown-model was None"),
        ):
            TextEmbedding.get_embedding_size("unknown-model")
