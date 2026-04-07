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
        for doc in documents:
            yield np.array([len(doc)], dtype=np.float32)

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        return super().get_embedding_size(model_name)

    @property
    def embedding_size(self) -> int:
        return super().embedding_size

    def token_count(self, texts: str | Iterable[str], **kwargs: Any) -> int:
        return super().token_count(texts, **kwargs)


def test_text_embedding_base_init():
    model_name = "test-model"
    cache_dir = "test-cache"
    threads = 4
    embedding = DummyTextEmbedding(
        model_name=model_name, cache_dir=cache_dir, threads=threads, local_files_only=True
    )

    assert embedding.model_name == model_name
    assert embedding.cache_dir == cache_dir
    assert embedding.threads == threads
    assert embedding._local_files_only is True


def test_passage_embed():
    embedding = DummyTextEmbedding(model_name="test")
    texts = ["hello", "world"]
    results = list(embedding.passage_embed(texts))

    assert len(results) == 2
    assert np.array_equal(results[0], np.array([5], dtype=np.float32))
    assert np.array_equal(results[1], np.array([5], dtype=np.float32))


def test_query_embed_single_string():
    embedding = DummyTextEmbedding(model_name="test")
    query = "hello"
    results = list(embedding.query_embed(query))

    assert len(results) == 1
    assert np.array_equal(results[0], np.array([5], dtype=np.float32))


def test_query_embed_iterable():
    embedding = DummyTextEmbedding(model_name="test")
    queries = ["hello", "world!"]
    results = list(embedding.query_embed(queries))

    assert len(results) == 2
    assert np.array_equal(results[0], np.array([5], dtype=np.float32))
    assert np.array_equal(results[1], np.array([6], dtype=np.float32))


def test_not_implemented_errors():
    embedding = DummyTextEmbedding(model_name="test")

    with pytest.raises(NotImplementedError):
        embedding.token_count("test")

    with pytest.raises(NotImplementedError):
        _ = embedding.embedding_size

    with pytest.raises(NotImplementedError):
        DummyTextEmbedding.get_embedding_size("test")


def test_base_embed_raises_not_implemented():
    class PureBase(TextEmbeddingBase):
        pass

    # We need to provide dummy args to init because TextEmbeddingBase.__init__ expects them
    base = PureBase(model_name="test")
    with pytest.raises(NotImplementedError):
        list(base.embed("test"))
