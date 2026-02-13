from collections.abc import Iterable
from typing import Any

from fastembed.common.model_description import DenseModelDescription
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.types import NumpyArray
from fastembed.common.utils import normalize
from fastembed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker
from fastembed.text.pooled_embedding import PooledEmbedding

# Base class model list kept empty — mean+normalize pooling models can
# be added at runtime via CustomTextEmbedding.add_model(pooling=PoolingType.MEAN, normalization=True).
supported_pooled_normalized_models: list[DenseModelDescription] = []


class PooledNormalizedEmbedding(PooledEmbedding):
    @classmethod
    def _get_worker_class(cls) -> type[OnnxTextEmbeddingWorker]:
        return PooledNormalizedEmbeddingWorker

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_pooled_normalized_models

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        if output.attention_mask is None:
            raise ValueError("attention_mask must be provided for document post-processing")

        embeddings = output.model_output
        attn_mask = output.attention_mask
        return normalize(self.mean_pooling(embeddings, attn_mask))


class PooledNormalizedEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextEmbedding:
        return PooledNormalizedEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
