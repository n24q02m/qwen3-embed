from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from qwen3_embed.common.types import NumpyArray
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


class DummyTextEmbedding(TextEmbeddingBase):
    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        if isinstance(documents, str):
            documents = [documents]
        for _ in documents:
            yield np.array([0.1, 0.2, 0.3])

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        return 3

    @property
    def embedding_size(self) -> int:
        return 3

    def token_count(self, texts: str | Iterable[str], **kwargs: Any) -> int:
        return 10

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        return []

    @classmethod
    def _list_supported_models(cls):
        return []


class MinimalTextEmbedding(TextEmbeddingBase):
    """Subclass that doesn't override abstract methods to test NotImplementedError."""

    pass


def test_text_embedding_base_init():
    model = DummyTextEmbedding(
        model_name="test-model", cache_dir="cache", threads=4, local_files_only=True
    )
    assert model.model_name == "test-model"
    assert model.cache_dir == "cache"
    assert model.threads == 4
    assert model._local_files_only is True


def test_text_embedding_base_passage_embed():
    model = DummyTextEmbedding(model_name="test-model")
    texts = ["hello", "world"]
    embeddings = list(model.passage_embed(texts))
    assert len(embeddings) == 2
    assert np.array_equal(embeddings[0], np.array([0.1, 0.2, 0.3]))


def test_text_embedding_base_query_embed_str():
    model = DummyTextEmbedding(model_name="test-model")
    query = "hello"
    embeddings = list(model.query_embed(query))
    assert len(embeddings) == 1
    assert np.array_equal(embeddings[0], np.array([0.1, 0.2, 0.3]))


def test_text_embedding_base_query_embed_iterable():
    model = DummyTextEmbedding(model_name="test-model")
    queries = ["hello", "world"]
    embeddings = list(model.query_embed(queries))
    assert len(embeddings) == 2
    assert np.array_equal(embeddings[0], np.array([0.1, 0.2, 0.3]))


def test_text_embedding_base_abstract_methods_raise():
    model = MinimalTextEmbedding(model_name="test-model")

    with pytest.raises(NotImplementedError):
        list(model.embed("test"))

    with pytest.raises(NotImplementedError):
        MinimalTextEmbedding.get_embedding_size("test-model")

    with pytest.raises(NotImplementedError):
        _ = model.embedding_size

    with pytest.raises(NotImplementedError):
        model.token_count("test")
