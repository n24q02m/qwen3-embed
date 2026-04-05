from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource, PoolingType
from qwen3_embed.common.types import Device, NumpyArray, OnnxProvider
from qwen3_embed.text.custom_text_embedding import CustomTextEmbedding
from qwen3_embed.text.gguf_embedding import Qwen3TextEmbeddingGGUF
from qwen3_embed.text.onnx_embedding import OnnxTextEmbedding
from qwen3_embed.text.pooled_embedding import PooledEmbedding
from qwen3_embed.text.pooled_normalized_embedding import PooledNormalizedEmbedding
from qwen3_embed.text.qwen3_embedding import Qwen3TextEmbedding
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


class TextEmbedding(TextEmbeddingBase):
    EMBEDDINGS_REGISTRY: list[type[TextEmbeddingBase]] = [
        Qwen3TextEmbedding,
        Qwen3TextEmbeddingGGUF,
        OnnxTextEmbedding,
        PooledNormalizedEmbedding,
        PooledEmbedding,
        CustomTextEmbedding,
    ]

    @classmethod
    def _get_embedding_caches(
        cls,
    ) -> tuple[dict[str, type[TextEmbeddingBase]], dict[str, DenseModelDescription]]:
        type_cache = vars(cls).get("_embedding_type_cache")
        desc_cache = vars(cls).get("_embedding_description_cache")
        if type_cache is None or desc_cache is None:
            # ⚡ Bolt: Implement O(1) dictionary caches for the embedding registry and model descriptions
            type_cache = {}
            desc_cache = {}
            for embedding_type in cls.EMBEDDINGS_REGISTRY:
                for model in embedding_type._list_supported_models():
                    m_lower = model.model.lower()
                    type_cache[m_lower] = embedding_type
                    desc_cache[m_lower] = model
            cls._embedding_type_cache = type_cache
            cls._embedding_description_cache = desc_cache
        return type_cache, desc_cache

    @classmethod
    def _clear_embedding_caches(cls) -> None:
        """Clears the embedding caches for the current class."""
        if "_embedding_type_cache" in vars(cls):
            delattr(cls, "_embedding_type_cache")
        if "_embedding_description_cache" in vars(cls):
            delattr(cls, "_embedding_description_cache")

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return [asdict(model) for model in cls._list_supported_models()]

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        _, desc_cache = cls._get_embedding_caches()
        return list(desc_cache.values())

    @classmethod
    def add_custom_model(
        cls,
        model: str,
        pooling: PoolingType,
        normalization: bool,
        sources: ModelSource,
        dim: int,
        model_file: str = "onnx/model.onnx",
        description: str = "",
        license: str = "",
        size_in_gb: float = 0.0,
        additional_files: list[str] | None = None,
    ) -> None:
        type_cache, _ = cls._get_embedding_caches()
        model_lower = model.lower()
        if model_lower in type_cache:
            raise ValueError(
                f"Model {model} is already registered in TextEmbedding, if you still want to add this model, "
                f"please use another model name"
            )

        CustomTextEmbedding.add_model(
            DenseModelDescription(
                model=model,
                sources=sources,
                dim=dim,
                model_file=model_file,
                description=description,
                license=license,
                size_in_GB=size_in_gb,
                additional_files=additional_files or [],
            ),
            pooling=pooling,
            normalization=normalization,
        )
        cls._clear_embedding_caches()

    def __init__(
        self,
        model_name: str = "n24q02m/Qwen3-Embedding-0.6B-ONNX",
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        type_cache, _ = self._get_embedding_caches()
        model_name_lower = model_name.lower()

        if model_name_lower in type_cache:
            embedding_type = type_cache[model_name_lower]
            self.model = embedding_type(
                model_name=model_name,
                cache_dir=cache_dir,
                threads=threads,
                providers=providers,
                cuda=cuda,
                device_ids=device_ids,
                lazy_load=lazy_load,
                **kwargs,
            )
            return

        raise ValueError(
            f"Model {model_name} is not supported in TextEmbedding. "
            "Please check the supported models using `TextEmbedding.list_supported_models()`"
        )

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the current model"""
        if self._embedding_size is None:
            self._embedding_size = self.get_embedding_size(self.model_name)
        return self._embedding_size

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        """Get the embedding size of the passed model

        Args:
            model_name (str): The name of the model to get embedding size for.

        Returns:
            int: The size of the embedding.

        Raises:
            ValueError: If the model name is not found in the supported models.
        """
        _, desc_cache = cls._get_embedding_caches()
        model_name_lower = model_name.lower()
        if model_name_lower in desc_cache:
            return desc_cache[model_name_lower].dim or 0

        model_names = list(desc_cache.keys())
        raise ValueError(
            f"Embedding size for model {model_name} was None. Available model names: {model_names}"
        )

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.

        Returns:
            List of embeddings, one per document
        """
        yield from self.model.embed(documents, batch_size, parallel, **kwargs)

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        """
        Embeds queries

        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable e.g. list of queries.

        Returns:
            Iterable[NumpyArray]: The embeddings.
        """
        # This is model-specific, so that different models can have specialized implementations
        yield from self.model.query_embed(query, **kwargs)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        """
        Embeds a list of text passages into a list of embeddings.

        Args:
            texts (Iterable[str]): The list of texts to embed.
            **kwargs: Additional keyword argument to pass to the embed method.

        Yields:
            Iterable[SparseEmbedding]: The sparse embeddings.
        """
        # This is model-specific, so that different models can have specialized implementations
        yield from self.model.passage_embed(texts, **kwargs)

    def token_count(
        self, texts: str | Iterable[str], batch_size: int = 1024, **kwargs: Any
    ) -> int:
        """Returns the number of tokens in the texts.

        Args:
            texts (str | Iterable[str]): The list of texts to embed.
            batch_size (int): Batch size for encoding

        Returns:
            int: Sum of number of tokens in the texts.
        """
        return self.model.token_count(texts, batch_size=batch_size, **kwargs)
