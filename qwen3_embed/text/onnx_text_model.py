import os
from collections.abc import Iterable
from multiprocessing import get_all_start_methods
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tokenizers import Encoding

from qwen3_embed.common.onnx_model import (
    EmbeddingWorker,
    OnnxInferenceConfig,
    OnnxModel,
    OnnxOutputContext,
    T,
)
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.common.utils import iter_batch
from qwen3_embed.parallel_processor import ParallelWorkerPool


class OnnxTextModel(OnnxModel[T]):
    ONNX_OUTPUT_NAMES: list[str] | None = None

    @classmethod
    def _get_worker_class(cls) -> type["TextEmbeddingWorker[T]"]:
        raise NotImplementedError("Subclasses must implement this method")

    def _post_process_onnx_output(self, output: OnnxOutputContext, **kwargs: Any) -> Iterable[T]:
        """Post-process the ONNX model output to convert it into a usable format.

        Args:
            output (OnnxOutputContext): The raw output from the ONNX model.
            **kwargs: Additional keyword arguments that may be needed by specific implementations.

        Returns:
            Iterable[T]: Post-processed output as an iterable of type T.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(self) -> None:
        super().__init__()

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray | NDArray[np.int64]]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    def load_onnx_model(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def tokenize(self, documents: list[str], **kwargs: Any) -> list[Encoding]:
        assert self.tokenizer is not None
        return self.tokenizer.encode_batch(documents)

    def onnx_embed(
        self,
        documents: list[str],
        **kwargs: Any,
    ) -> OnnxOutputContext:
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_onnx_model() first.")
        encoded = self.tokenize(documents, **kwargs)
        input_ids = np.array([e.ids for e in encoded])
        attention_mask = np.array([e.attention_mask for e in encoded])
        input_names = self.model_input_names or set()
        assert input_names is not None
        onnx_input: dict[str, NumpyArray] = {
            "input_ids": np.array(input_ids, dtype=np.int64),
        }
        if "attention_mask" in input_names:
            onnx_input["attention_mask"] = np.array(attention_mask, dtype=np.int64)
        if "token_type_ids" in input_names:
            onnx_input["token_type_ids"] = np.zeros(input_ids.shape, dtype=np.int64)
        onnx_input = self._preprocess_onnx_input(onnx_input, **kwargs)

        model_output = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)
        result = model_output[0]
        if getattr(result, "dtype", None) == np.float16:
            result = result.astype(np.float32)  # type: ignore[unresolved-attribute]
        return OnnxOutputContext(
            model_output=result,  # type: ignore[invalid-argument-type]
            attention_mask=onnx_input.get("attention_mask", attention_mask),
            input_ids=onnx_input.get("input_ids", input_ids),
        )

    def _embed_documents(
        self,
        documents: str | Iterable[str],
        config: OnnxInferenceConfig,
        **kwargs: Any,
    ) -> Iterable[T]:
        is_small = False

        if isinstance(documents, str):
            documents = [documents]
            is_small = True

        if isinstance(documents, list) and len(documents) < config.batch_size:
            is_small = True

        if config.parallel is None or is_small:
            if not hasattr(self, "model") or self.model is None:
                self.load_onnx_model()
            for batch in iter_batch(documents, config.batch_size):
                yield from self._post_process_onnx_output(
                    self.onnx_embed(batch, **kwargs), **kwargs
                )
        else:
            parallel = config.parallel
            if parallel == 0:
                parallel = os.cpu_count()

            start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
            params = {
                "model_name": config.model_name,
                "cache_dir": config.cache_dir,
                "providers": config.providers,
                "local_files_only": config.local_files_only,
                "specific_model_path": config.specific_model_path,
                **kwargs,
            }

            if config.extra_session_options is not None:
                params.update(config.extra_session_options)

            pool = ParallelWorkerPool(
                num_workers=parallel or 1,
                worker=self._get_worker_class(),
                cuda=config.cuda,
                device_ids=config.device_ids,
                start_method=start_method,
            )
            for batch in pool.ordered_map(iter_batch(documents, config.batch_size), **params):
                yield from self._post_process_onnx_output(batch, **kwargs)

    def _token_count(self, texts: str | Iterable[str], batch_size: int = 1024, **_: Any) -> int:
        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()  # loads the tokenizer as well

        token_num = 0
        assert self.tokenizer is not None
        texts = [texts] if isinstance(texts, str) else texts
        for batch in iter_batch(texts, batch_size):
            for tokens in self.tokenizer.encode_batch(batch):
                # ⚡ Bolt: Fast token counting using .count(1) (~30% faster than sum())
                token_num += tokens.attention_mask.count(1)

        return token_num


class TextEmbeddingWorker(EmbeddingWorker[T]):
    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, OnnxOutputContext]]:
        for idx, batch in items:
            onnx_output = self.model.onnx_embed(batch)
            yield idx, onnx_output
