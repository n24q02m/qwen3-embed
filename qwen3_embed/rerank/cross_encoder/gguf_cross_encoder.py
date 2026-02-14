"""Qwen3 reranker via GGUF (llama-cpp-python) with yes/no logit scoring.

Runtime alternative to ONNX for Qwen3-Reranker. Uses llama-cpp-python for
inference from GGUF quantized models (Q4_K_M).

Requires optional dependency: pip install qwen3-embed[gguf]
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qwen3_embed.common.model_description import BaseModelDescription, ModelSource
from qwen3_embed.common.types import Device, NumpyArray, OnnxProvider
from qwen3_embed.common.utils import define_cache_dir
from qwen3_embed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase

# ---------------------------------------------------------------------------
# Qwen3 reranker constants (same as ONNX version)
# ---------------------------------------------------------------------------
TOKEN_YES_ID = 9693
TOKEN_NO_ID = 2132

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)

DEFAULT_INSTRUCTION = (
    "Given a query and a document, judge whether the document is relevant to the query."
)

RERANK_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n<Instruct>: {instruction}\n"
    "<Query>: {query}\n<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
supported_qwen3_reranker_gguf_models: list[BaseModelDescription] = [
    BaseModelDescription(
        model="Qwen/Qwen3-Reranker-0.6B-GGUF",
        description=(
            "Qwen3 reranker (0.6B) using causal LM yes/no scoring. "
            "GGUF Q4_K_M quantized. Multilingual, 40960 input tokens, "
            "instruction-aware, 2025 year."
        ),
        license="apache-2.0",
        size_in_GB=0.48,
        sources=ModelSource(hf="n24q02m/Qwen3-Reranker-0.6B-ONNX"),
        model_file="gguf/qwen3-reranker-0.6b-q4-k-m.gguf",
    ),
]


def _check_llama_cpp() -> None:
    """Check that llama-cpp-python is installed."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError as e:
        msg = (
            "llama-cpp-python is required for GGUF models. "
            "Install with: pip install qwen3-embed[gguf]"
        )
        raise ImportError(msg) from e


# ---------------------------------------------------------------------------
# GGUF reranker implementation
# ---------------------------------------------------------------------------
class Qwen3CrossEncoderGGUF(TextCrossEncoderBase):
    """Qwen3 GGUF Reranker using causal LM with yes/no logit scoring.

    Uses llama-cpp-python for inference. Compared to ONNX variants:
    - Smaller model files (Q4_K_M ~480MB vs INT8 ~573MB)
    - Slightly higher latency on CPU

    Usage::

        from qwen3_embed import TextCrossEncoder

        reranker = TextCrossEncoder("Qwen/Qwen3-Reranker-0.6B-GGUF")
        scores = list(reranker.rerank("What is AI?", ["doc1", "doc2"]))
    """

    @classmethod
    def _list_supported_models(cls) -> list[BaseModelDescription]:
        return supported_qwen3_reranker_gguf_models

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B-GGUF",
        cache_dir: str | None = None,
        threads: int | None = None,
        # Accept but ignore ONNX-specific args for compatibility with TextCrossEncoder dispatcher
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ) -> None:
        _check_llama_cpp()
        super().__init__(model_name, cache_dir, threads, **kwargs)

        self.model_description = self._get_model_description(model_name)
        self.cache_dir = str(define_cache_dir(cache_dir))

        self._model_dir = self.download_model(
            self.model_description,
            self.cache_dir,
            local_files_only=self._local_files_only,
        )

        model_path = Path(self._model_dir) / self.model_description.model_file
        if not model_path.exists():
            msg = f"GGUF model file not found: {model_path}"
            raise FileNotFoundError(msg)

        from llama_cpp import Llama

        n_gpu = -1 if (cuda is True or cuda == Device.CUDA) else 0
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=40960,
            logits_all=False,  # Only need last-token logits
            n_threads=threads or 0,
            n_gpu_layers=n_gpu,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Chat template formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format_rerank_input(
        query: str,
        document: str,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> str:
        """Build the chat-template string for a single query-document pair."""
        return RERANK_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            instruction=instruction,
            query=query,
            document=document,
        )

    # ------------------------------------------------------------------
    # Yes/No logit scoring
    # ------------------------------------------------------------------
    def _score_text(self, text: str) -> float:
        """Tokenize text, evaluate, and extract P(yes) from last-token logits."""
        tokens = self._llm.tokenize(text.encode("utf-8"), add_bos=False)

        self._llm.reset()
        self._llm.eval(tokens)

        # Get logits for the last token
        # llama-cpp-python stores scores in _scores array
        last_logits: NumpyArray = np.array(self._llm.scores[len(tokens) - 1], dtype=np.float32)

        # Extract yes/no logits
        yes_logit = last_logits[TOKEN_YES_ID]
        no_logit = last_logits[TOKEN_NO_ID]

        # Numerically stable softmax over [no, yes]
        max_logit = max(float(yes_logit), float(no_logit))
        exp_yes = np.exp(yes_logit - max_logit)
        exp_no = np.exp(no_logit - max_logit)
        p_yes = float(exp_yes / (exp_yes + exp_no))

        return p_yes

    # ------------------------------------------------------------------
    # rerank / rerank_pairs
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        batch_size: int = 64,
        **kwargs: Any,
    ) -> Iterable[float]:
        """Rerank documents based on relevance to a query.

        Args:
            query: The query string.
            documents: Iterable of documents to rerank.
            batch_size: Ignored (processes one at a time).
            **kwargs: ``instruction`` (str) overrides default instruction.

        Yields:
            float: Relevance scores P(yes) for each document.
        """
        instruction = kwargs.pop("instruction", DEFAULT_INSTRUCTION)
        for doc in documents:
            text = self._format_rerank_input(query, doc, instruction)
            yield self._score_text(text)

    def rerank_pairs(
        self,
        pairs: Iterable[tuple[str, str]],
        batch_size: int = 64,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[float]:
        """Rerank pre-formed (query, document) pairs.

        Args:
            pairs: Iterable of (query, document) tuples.
            batch_size: Ignored.
            parallel: Ignored.
            **kwargs: ``instruction`` (str) overrides default instruction.

        Yields:
            float: Relevance scores P(yes) for each pair.
        """
        instruction = kwargs.pop("instruction", DEFAULT_INSTRUCTION)
        for query, doc in pairs:
            text = self._format_rerank_input(query, doc, instruction)
            yield self._score_text(text)
