from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

from qwen3_embed.common import OnnxProvider
from qwen3_embed.common.model_description import (
    BaseModelDescription,
    ModelSource,
)
from qwen3_embed.common.types import Device
from qwen3_embed.rerank.cross_encoder.custom_text_cross_encoder import CustomTextCrossEncoder
from qwen3_embed.rerank.cross_encoder.gguf_cross_encoder import Qwen3CrossEncoderGGUF
from qwen3_embed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder
from qwen3_embed.rerank.cross_encoder.qwen3_cross_encoder import Qwen3CrossEncoder
from qwen3_embed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase


class TextCrossEncoder(TextCrossEncoderBase):
    CROSS_ENCODER_REGISTRY: list[type[TextCrossEncoderBase]] = [
        Qwen3CrossEncoder,
        Qwen3CrossEncoderGGUF,
        OnnxTextCrossEncoder,
        CustomTextCrossEncoder,
    ]
    _encoder_type_cache: dict[str, type[TextCrossEncoderBase]] | None = None
    _encoder_description_cache: dict[str, BaseModelDescription] | None = None

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[BaseModelDescription]: A list of dictionaries containing the model information.

            Example:
                ```
                [
                    {
                        "model": "Xenova/ms-marco-MiniLM-L-6-v2",
                        "size_in_GB": 0.08,
                        "sources": {
                            "hf": "Xenova/ms-marco-MiniLM-L-6-v2",
                        },
                        "model_file": "onnx/model.onnx",
                        "description": "MiniLM-L-6-v2 model optimized for re-ranking tasks.",
                        "license": "apache-2.0",
                    }
                ]
                ```
        """
        return [asdict(model) for model in cls._list_supported_models()]

    @classmethod
    def _list_supported_models(cls) -> list[BaseModelDescription]:
        # Bolt: Use __dict__.get to ensure we use the cache for THIS specific class.
        cache = cls.__dict__.get("_encoder_description_cache")
        if cache is None:
            # Cache model descriptions to avoid repeated registry scans
            # and repeated lowercase operations. We use a dictionary to maintain
            # precedence (first match in registry wins) by iterating in reverse.
            descriptions = []
            for encoder in cls.CROSS_ENCODER_REGISTRY:
                descriptions.extend(encoder._list_supported_models())

            cache = {}
            for desc in reversed(descriptions):
                cache[desc.model.lower()] = desc
            cls._encoder_description_cache = cache

        return list(cache.values())

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)

        # Bolt: Use __dict__.get on the class to avoid inheritance issues.
        cache = self.__class__.__dict__.get("_encoder_type_cache")
        if cache is None:
            # Cache model-to-type mapping for O(1) dispatch
            cache = {}
            for CROSS_ENCODER_TYPE in reversed(self.CROSS_ENCODER_REGISTRY):
                for model in CROSS_ENCODER_TYPE._list_supported_models():
                    cache[model.model.lower()] = CROSS_ENCODER_TYPE
            self.__class__._encoder_type_cache = cache

        model_name_lower = model_name.lower()
        if model_name_lower in cache:
            CROSS_ENCODER_TYPE = cache[model_name_lower]
            self.model = CROSS_ENCODER_TYPE(
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
            f"Model {model_name} is not supported in TextCrossEncoder."
            "Please check the supported models using `TextCrossEncoder.list_supported_models()`"
        )

    def rerank(
        self, query: str, documents: Iterable[str], batch_size: int = 64, **kwargs: Any
    ) -> Iterable[float]:
        """Rerank a list of documents based on a query.

        Args:
            query: Query to rerank the documents against
            documents: Iterator of documents to rerank
            batch_size: Batch size for reranking

        Returns:
            Iterable of scores for each document
        """
        from qwen3_embed.common.utils import check_input_length, iter_checked_texts

        check_input_length(query)
        docs = iter_checked_texts(documents)

        yield from self.model.rerank(query, docs, batch_size=batch_size, **kwargs)

    def rerank_pairs(
        self,
        pairs: Iterable[tuple[str, str]],
        batch_size: int = 64,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[float]:
        """
        Rerank a list of query-document pairs.

        Args:
            pairs (Iterable[tuple[str, str]]): An iterable of tuples, where each tuple contains a query and a document
                to be scored together.
            batch_size (int, optional): The number of query-document pairs to process in a single batch. Defaults to 64.
            parallel (Optional[int], optional): The number of parallel processes to use for reranking.
                If None, parallelization is disabled. Defaults to None.
            **kwargs (Any): Additional arguments to pass to the underlying reranking model.

        Returns:
            Iterable[float]: An iterable of scores corresponding to each query-document pair in the input.
            Higher scores indicate a stronger match between the query and the document.

        Example:
            >>> encoder = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
            >>> pairs = [("What is AI?", "Artificial intelligence is ..."), ("What is ML?", "Machine learning is ...")]
            >>> scores = list(encoder.rerank_pairs(pairs))
            >>> print(list(map(lambda x: round(x, 2), scores)))
            [-1.24, -10.6]
        """
        from qwen3_embed.common.utils import check_input_length

        def _check_pairs(ps):
            for q, d in ps:
                check_input_length(q)
                check_input_length(d)
                yield q, d

        yield from self.model.rerank_pairs(
            _check_pairs(pairs), batch_size=batch_size, parallel=parallel, **kwargs
        )

    @classmethod
    def add_custom_model(
        cls,
        model: str,
        sources: ModelSource,
        model_file: str = "onnx/model.onnx",
        description: str = "",
        license: str = "",
        size_in_gb: float = 0.0,
        additional_files: list[str] | None = None,
    ) -> None:
        # Bolt: Use O(1) cache for duplicate check
        cls._list_supported_models()
        assert cls._encoder_description_cache is not None

        model_lower = model.lower()
        if model_lower in cls._encoder_description_cache:
            raise ValueError(
                f"Model {model} is already registered in CrossEncoderModel, if you still want to add this model, "
                f"please use another model name"
            )

        CustomTextCrossEncoder.add_model(
            BaseModelDescription(
                model=model,
                sources=sources,
                model_file=model_file,
                description=description,
                license=license,
                size_in_GB=size_in_gb,
                additional_files=additional_files or [],
            )
        )
        cls._clear_caches()

    @classmethod
    def _clear_caches(cls) -> None:
        cls._encoder_type_cache = None
        cls._encoder_description_cache = None

    def token_count(
        self, pairs: Iterable[tuple[str, str]], batch_size: int = 1024, **kwargs: Any
    ) -> int:
        """Returns the number of tokens in the pairs.

        Args:
            pairs: Iterable of tuples, where each tuple contains a query and a document to be tokenized
            batch_size: Batch size for tokenizing

        Returns:
            token count: overall number of tokens in the pairs
        """
        return self.model.token_count(pairs, batch_size=batch_size, **kwargs)
