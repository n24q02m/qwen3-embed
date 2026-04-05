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
            yield np.zeros(10)

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        return 10

    @property
    def embedding_size(self) -> int:
        return 10

    def token_count(self, texts: str | Iterable[str], **kwargs: Any) -> int:
        if isinstance(texts, str):
            return len(texts.split())
        return sum(len(t.split()) for t in texts)

    @classmethod
    def _list_supported_models(cls) -> list[Any]:
        return []


def test_text_embedding_base_init():
    model_name = "test-model"
    cache_dir = "test-cache"
    threads = 4
    local_files_only = True

    embedding = DummyTextEmbedding(
        model_name=model_name,
        cache_dir=cache_dir,
        threads=threads,
        local_files_only=local_files_only,
        extra_arg="extra",
    )

    assert embedding.model_name == model_name
    assert embedding.cache_dir == cache_dir
    assert embedding.threads == threads
    assert embedding._local_files_only is local_files_only


def test_passage_embed():
    embedding = DummyTextEmbedding(model_name="test")
    texts = ["hello", "world"]
    results = list(embedding.passage_embed(texts))

    assert len(results) == 2
    assert all(isinstance(res, np.ndarray) for res in results)
    assert all(res.shape == (10,) for res in results)


def test_query_embed_single():
    embedding = DummyTextEmbedding(model_name="test")
    query = "what is test?"
    results = list(embedding.query_embed(query))

    assert len(results) == 1
    assert isinstance(results[0], np.ndarray)
    assert results[0].shape == (10,)


def test_query_embed_multiple():
    embedding = DummyTextEmbedding(model_name="test")
    queries = ["q1", "q2"]
    results = list(embedding.query_embed(queries))

    assert len(results) == 2
    assert all(isinstance(res, np.ndarray) for res in results)
    assert all(res.shape == (10,) for res in results)


def test_not_implemented_methods():
    class MinimalTextEmbedding(TextEmbeddingBase):
        @classmethod
        def _list_supported_models(cls) -> list[Any]:
            return []

    embedding = MinimalTextEmbedding(model_name="test")

    with pytest.raises(NotImplementedError):
        list(embedding.embed("test"))

    with pytest.raises(NotImplementedError):
        MinimalTextEmbedding.get_embedding_size("test")

    with pytest.raises(NotImplementedError):
        _ = embedding.embedding_size

    with pytest.raises(NotImplementedError):
        embedding.token_count("test")
