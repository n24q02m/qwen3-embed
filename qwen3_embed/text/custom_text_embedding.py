from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qwen3_embed.common.model_description import (
    DenseModelDescription,
    PoolingType,
)
from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.common.utils import last_token_pool, mean_pooling, normalize
from qwen3_embed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker


@dataclass(frozen=True)
class PostprocessingConfig:
    pooling: PoolingType
    normalization: bool


class CustomTextEmbedding(OnnxTextEmbedding):
    SUPPORTED_MODELS: list[DenseModelDescription] = []
    POSTPROCESSING_MAPPING: dict[str, PostprocessingConfig] = {}

    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            **kwargs,
        )
        self._pooling = self.POSTPROCESSING_MAPPING[model_name].pooling
        self._normalization = self.POSTPROCESSING_MAPPING[model_name].normalization

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return cls.SUPPORTED_MODELS

    @classmethod
    def _get_worker_class(cls) -> type["CustomTextEmbeddingWorker"]:
        return CustomTextEmbeddingWorker

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        embeddings = self._pool(output.model_output, output.attention_mask)

        # MRL: optionally truncate to requested dimension
        dim: int | None = kwargs.get("dim")
        if dim is not None:
            embeddings = embeddings[:, :dim]

        return self._normalize(embeddings)

    def _pool(
        self, embeddings: NumpyArray, attention_mask: NDArray[np.int64] | None = None
    ) -> NumpyArray:
        if self._pooling == PoolingType.CLS:
            return embeddings[:, 0]

        if self._pooling == PoolingType.MEAN:
            if attention_mask is None:
                raise ValueError("attention_mask must be provided for mean pooling")
            return mean_pooling(embeddings, attention_mask)

        if self._pooling == PoolingType.LAST_TOKEN:
            if attention_mask is None:
                raise ValueError("attention_mask must be provided for last-token pooling")
            return last_token_pool(embeddings, attention_mask)

        if self._pooling == PoolingType.DISABLED:
            return embeddings

        raise ValueError(
            f"Unsupported pooling type {self._pooling}. "
            f"Supported types are: {PoolingType.CLS}, {PoolingType.MEAN}, "
            f"{PoolingType.LAST_TOKEN}, {PoolingType.DISABLED}."
        )

    def _normalize(self, embeddings: NumpyArray) -> NumpyArray:
        return normalize(embeddings) if self._normalization else embeddings

    @classmethod
    def add_model(
        cls,
        model_description: DenseModelDescription,
        pooling: PoolingType,
        normalization: bool,
    ) -> None:
        cls._clear_model_cache()
        cls.SUPPORTED_MODELS.append(model_description)
        cls.POSTPROCESSING_MAPPING[model_description.model] = PostprocessingConfig(
            pooling=pooling, normalization=normalization
        )


class CustomTextEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> CustomTextEmbedding:
        return CustomTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
