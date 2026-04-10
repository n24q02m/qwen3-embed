from collections.abc import Iterable
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.text.text_embedding import TextEmbedding
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


class MockEmbedding(TextEmbeddingBase):
    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self.query_embed_calls = []

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return [
            DenseModelDescription(
                model="mock-model",
                dim=128,
                description="Mock model for testing",
                sources=ModelSource(hf="mock/model"),
                model_file="model.onnx",
                license="Apache-2.0",
                size_in_GB=0.1,
            )
        ]

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        self.query_embed_calls.append((query, kwargs))
        yield np.zeros(128)


def test_query_embed_delegation():
    """Verify that TextEmbedding.query_embed correctly delegates to the underlying model."""
    with patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [MockEmbedding]):
        model = TextEmbedding(model_name="mock-model")
        assert isinstance(model.model, MockEmbedding)

        # Test single string
        results = list(model.query_embed("hello", some_arg="value"))
        assert len(results) == 1
        assert results[0].shape == (128,)
        assert model.model.query_embed_calls[0][0] == "hello"
        assert model.model.query_embed_calls[0][1] == {"some_arg": "value"}

        # Test iterable
        results = list(model.query_embed(["hello", "world"]))
        assert len(results) == 1
        query_arg = model.model.query_embed_calls[1][0]
        assert isinstance(query_arg, Iterable) and not isinstance(query_arg, (str, list, tuple))
        assert list(query_arg) == ["hello", "world"]


def test_query_embed_length_validation():
    """Verify that TextEmbedding.query_embed validates input length."""
    with patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [MockEmbedding]):
        model = TextEmbedding(model_name="mock-model")

        # Test with extremely long string
        long_string = "a" * 1000001  # Exceeds default MAX_INPUT_LENGTH
        with pytest.raises(ValueError, match="exceeds maximum allowed length"):
            list(model.query_embed(long_string))


class MockIteratingEmbedding(MockEmbedding):
    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        if isinstance(query, str):
            yield np.zeros(128)
        else:
            for _ in query:
                yield np.zeros(128)


def test_query_embed_length_validation_iterative():
    with patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [MockIteratingEmbedding]):
        model = TextEmbedding(model_name="mock-model")
        long_string = "a" * 1000001

        # Test with iterable containing a long string
        gen = model.query_embed(["valid", long_string])
        assert next(gen) is not None  # "valid" passes
        with pytest.raises(ValueError, match="exceeds maximum allowed length"):
            next(gen)  # long_string fails
