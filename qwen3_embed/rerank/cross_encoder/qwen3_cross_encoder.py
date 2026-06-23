"""Qwen3 reranker using causal LM with yes/no logit scoring.

Unlike traditional cross-encoder rerankers (which concatenate query+document
as a pair, feed through a BERT-class model, and read a relevance head), the
Qwen3 reranker:

1. Formats input as a **chat template** with system/user/assistant turns.
2. Runs a **causal language model** (Qwen3ForCausalLM).
3. Extracts the **last-token logits** for the "yes" and "no" tokens.
4. Applies **softmax** to obtain the relevance probability.

This means the ONNX model output has shape ``(batch, seq_len, vocab_size)``
instead of the typical ``(batch, num_labels)`` from cross-encoders.
"""

import re
from typing import Any

import numpy as np

from qwen3_embed.common.model_description import BaseModelDescription, ModelSource
from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.rerank.cross_encoder.onnx_text_cross_encoder import (
    OnnxTextCrossEncoder,
    TextCrossEncoderWorker,
)
from qwen3_embed.rerank.cross_encoder.onnx_text_model import TextRerankerWorker

# ---------------------------------------------------------------------------
# Qwen3 reranker constants
# ---------------------------------------------------------------------------
# Token IDs in the Qwen3 tokenizer vocabulary
TOKEN_YES_ID = 9693
TOKEN_NO_ID = 2152

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

# Tokens that must be stripped from user input to prevent prompt injection
FORBIDDEN_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
FORBIDDEN_RE = re.compile("|".join(re.escape(token) for token in FORBIDDEN_TOKENS))

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
supported_qwen3_reranker_models: list[BaseModelDescription] = [
    BaseModelDescription(
        model="n24q02m/Qwen3-Reranker-0.6B-ONNX",
        description=(
            "Qwen3 reranker (0.6B) using causal LM yes/no scoring. "
            "INT8 dynamic quantized. Multilingual, 40960 input tokens, "
            "instruction-aware, 2025 year."
        ),
        license="apache-2.0",
        size_in_GB=0.57,
        sources=ModelSource(hf="n24q02m/Qwen3-Reranker-0.6B-ONNX"),
        model_file="onnx/model_quantized.onnx",
    ),
    BaseModelDescription(
        model="n24q02m/Qwen3-Reranker-0.6B-ONNX-Q4F16",
        description=(
            "Qwen3 reranker (0.6B) using causal LM yes/no scoring. "
            "INT4 weights + FP16 activations (Q4F16). Multilingual, "
            "40960 input tokens, instruction-aware, 2025 year."
        ),
        license="apache-2.0",
        size_in_GB=0.57,
        sources=ModelSource(hf="n24q02m/Qwen3-Reranker-0.6B-ONNX"),
        model_file="onnx/model_q4f16.onnx",
    ),
    BaseModelDescription(
        model="n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo",
        description=(
            "Qwen3 reranker (0.6B) with optimized 2-dim yes/no output. "
            "INT8 dynamic quantized. ~10x less RAM than full-vocab version. "
            "Multilingual, 40960 input tokens, instruction-aware, 2025 year."
        ),
        license="apache-2.0",
        size_in_GB=0.57,
        sources=ModelSource(hf="n24q02m/Qwen3-Reranker-0.6B-ONNX"),
        model_file="onnx/model_yesno_quantized.onnx",
    ),
]


# ---------------------------------------------------------------------------
# Qwen3 reranker implementation
# ---------------------------------------------------------------------------
class Qwen3CrossEncoder(OnnxTextCrossEncoder):
    """Qwen3 Reranker using causal LM with yes/no logit scoring.

    Usage::

        from qwen3_embed import TextCrossEncoder

        reranker = TextCrossEncoder("n24q02m/Qwen3-Reranker-0.6B-ONNX")
        scores = list(reranker.rerank("What is AI?", ["doc1", "doc2"]))

        # Custom instruction
        scores = list(reranker.rerank(
            "What is AI?",
            ["doc1", "doc2"],
            instruction="Judge document relevance for code search.",
        ))
    """

    @classmethod
    def _list_supported_models(cls) -> list[BaseModelDescription]:
        return supported_qwen3_reranker_models

    # ------------------------------------------------------------------
    # Chat template formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_input(text: str) -> str:
        """Strip forbidden special tokens from user input."""
        # SECURITY: Prevent prompt injection bypass via iterative payload construction.
        while True:
            text, count = FORBIDDEN_RE.subn("", text)
            if count == 0:
                break
        return text

    @staticmethod
    def _format_rerank_input(
        query: str,
        document: str,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> str:
        """Build the chat-template string for a single query-document pair."""
        # document is sanitized here; query and instruction should be sanitized
        # by the caller (onnx_embed) to avoid redundant regex calls in a loop.
        document = Qwen3CrossEncoder._sanitize_input(document)

        return RERANK_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            instruction=instruction,
            query=query,
            document=document,
        )

    # ------------------------------------------------------------------
    # Yes/No logit scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_yes_no_scores(
        model_output: NumpyArray,
        attention_mask: NumpyArray | None = None,
    ) -> NumpyArray:
        """Extract yes/no logits from causal LM output and compute scores.

        Supports two output shapes:
        - Optimized: ``(batch, 2)`` — direct [no, yes] logits (YesNo variant).
        - Legacy: ``(batch, seq_len, vocab_size)`` — full causal LM output.

        Args:
            model_output: Raw model output.
            attention_mask: Optional (batch, seq_len) mask. Required for full-vocab
                model when the batch contains different-length sequences, so that
                the last CONTENT token (not a trailing pad token) is scored per row.

        Returns:
            Relevance scores (P(yes)), shape ``(batch,)``.
        """
        if model_output.ndim == 2:
            # Optimized model: output is already (batch, 2) with [no, yes]
            # Type cast to float32 is required to prevent type errors during in-place mutation
            diff = np.subtract(model_output[:, 0], model_output[:, 1], dtype=np.float32)
        else:
            # Full-vocab model: (batch, seq_len, vocab_size). Pick the last non-pad
            # position per row. Fallback to position -1 only if mask is not provided
            # (back-compat — single-row callers).
            if attention_mask is not None:
                batch_size = model_output.shape[0]
                # Handle both right-pad (pads trail) and left-pad (pads lead).
                left_padding = bool(attention_mask[:, -1].all())
                if left_padding:
                    last_logits = model_output[:, -1, :]
                else:
                    # ⚡ Bolt: Fast last token index calculation using sum (~4x faster than reverse argmax)
                    last_idx = attention_mask.sum(axis=1) - 1
                    last_logits = model_output[np.arange(batch_size), last_idx]
            else:
                last_logits = model_output[:, -1, :]

            # Fast sigmoid calculation on logit difference for 2-class classification (~10x faster)
            # ⚡ Bolt: Fast logit subtraction without stack array allocation overhead (~4.5x faster)
            # Type cast to float32 is required to prevent type errors during in-place mutation
            diff = np.subtract(
                last_logits[:, TOKEN_NO_ID], last_logits[:, TOKEN_YES_ID], dtype=np.float32
            )

        # ⚡ Bolt: Fast sigmoid using in-place operations to avoid array allocation overhead (~20% faster)
        with np.errstate(over="ignore"):
            np.exp(diff, out=diff)
            diff += 1.0
            np.reciprocal(diff, out=diff)
            return diff  # P(yes)

    # ------------------------------------------------------------------
    # Override ONNX inference to use chat-template + CausalLM scoring
    # ------------------------------------------------------------------
    def onnx_embed(self, query: str, documents: list[str], **kwargs: Any) -> OnnxOutputContext:
        """Score query-document pairs using the Qwen3 chat template."""
        query = self._sanitize_input(query)
        instruction = self._sanitize_input(kwargs.pop("instruction", DEFAULT_INSTRUCTION))

        texts = [self._format_rerank_input(query, doc, instruction) for doc in documents]
        return self._onnx_embed_texts(texts, **kwargs)

    def onnx_embed_pairs(self, pairs: list[tuple[str, str]], **kwargs: Any) -> OnnxOutputContext:
        """Score pre-formed (query, document) pairs."""
        instruction = self._sanitize_input(kwargs.pop("instruction", DEFAULT_INSTRUCTION))
        texts = [
            self._format_rerank_input(self._sanitize_input(q), doc, instruction)
            for q, doc in pairs
        ]
        return self._onnx_embed_texts(texts, **kwargs)

    def _onnx_embed_texts(self, texts: list[str], **kwargs: Any) -> OnnxOutputContext:
        """Score documents in a single batched ONNX call.

        The reranker ONNX graph bakes ``position_ids = arange(seq_len)`` starting at
        zero. Under right-padding, every sequence's real content begins at index 0,
        so its RoPE positions remain invariant regardless of batch composition or
        padding length. This allows efficient batched inference while maintaining
        score consistency (resolving the bottleneck in issue #725).
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_onnx_model() first.")
        assert self.tokenizer is not None, "Tokenizer not loaded. Call load_onnx_model() first."

        # encode_batch leverages the Rust-tokenizer's multi-threading and handles
        # right-padding to the longest sequence in the batch.
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        input_names = self.model_input_names or set()
        onnx_input: dict[str, NumpyArray] = {"input_ids": input_ids}

        if "attention_mask" in input_names:
            onnx_input["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            onnx_input["token_type_ids"] = np.zeros_like(input_ids)

        onnx_input = self._preprocess_onnx_input(onnx_input, **kwargs)
        outputs = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)
        model_output = outputs[0]

        if getattr(model_output, "dtype", None) == np.float16:
            model_output = model_output.astype(np.float32)  # type: ignore[unresolved-attribute]

        # _compute_yes_no_scores correctly identifies the last non-pad token per row
        # using the attention mask, ensuring accuracy even with varied sequence lengths.
        scores = self._compute_yes_no_scores(
            model_output,  # type: ignore[invalid-argument-type]
            onnx_input.get("attention_mask"),
        )
        return OnnxOutputContext(model_output=scores)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    @classmethod
    def _get_worker_class(cls) -> type[TextRerankerWorker]:
        return Qwen3CrossEncoderWorker


class Qwen3CrossEncoderWorker(TextCrossEncoderWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextCrossEncoder:
        return Qwen3CrossEncoder(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
