from collections.abc import Iterable
from typing import Any

import numpy as np

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


class MockEmbeddingModel(TextEmbeddingBase):
    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return [
            DenseModelDescription(
                model="mock-model",
                dim=128,
                description="Mock model for testing",
                license="MIT",
                sources=ModelSource(hf="mock/mock-model"),
                model_file="model.onnx",
                size_in_GB=0.1,
            )
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        self.calls.append(("embed", documents, batch_size, parallel, kwargs))
        yield np.zeros(128)

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        self.calls.append(("query_embed", query, kwargs))
        yield np.zeros(128)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        self.calls.append(("passage_embed", texts, kwargs))
        yield np.zeros(128)

    def token_count(
        self, texts: str | Iterable[str], batch_size: int = 1024, **kwargs: Any
    ) -> int:
        self.calls.append(("token_count", texts, {"batch_size": batch_size, **kwargs}))
        return 42
