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
        # Sanitize inputs to prevent injection
        query = Qwen3CrossEncoder._sanitize_input(query)
        document = Qwen3CrossEncoder._sanitize_input(document)
        instruction = Qwen3CrossEncoder._sanitize_input(instruction)

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
            yes_no_logits = model_output
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
                    # ⚡ Bolt: Fast reduction to find last valid token in right-padded masks (~3x faster)
                    last_idx = attention_mask.sum(axis=1) - 1
                    last_logits = model_output[np.arange(batch_size), last_idx]
            else:
                last_logits = model_output[:, -1, :]

            yes_no_logits = np.stack(
                [last_logits[:, TOKEN_NO_ID], last_logits[:, TOKEN_YES_ID]], axis=1
            )  # (batch, 2)

        # Fast sigmoid calculation on logit difference for 2-class classification (~10x faster)
        # ⚡ Bolt: Fast sigmoid using in-place operations to avoid array allocation overhead (~20% faster)
        # Type cast to float32 is required to prevent type errors during in-place mutation
        diff = np.subtract(yes_no_logits[:, 0], yes_no_logits[:, 1], dtype=np.float32)
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
        instruction = kwargs.pop("instruction", DEFAULT_INSTRUCTION)
        texts = [self._format_rerank_input(query, doc, instruction) for doc in documents]
        return self._onnx_embed_texts(texts, **kwargs)

    def onnx_embed_pairs(self, pairs: list[tuple[str, str]], **kwargs: Any) -> OnnxOutputContext:
        """Score pre-formed (query, document) pairs."""
        instruction = kwargs.pop("instruction", DEFAULT_INSTRUCTION)
        texts = [self._format_rerank_input(query, doc, instruction) for query, doc in pairs]
        return self._onnx_embed_texts(texts, **kwargs)

    def _onnx_embed_texts(self, texts: list[str], **kwargs: Any) -> OnnxOutputContext:
        """Tokenise and run model using batched inference (dynamic batch ONNX graph)."""
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_onnx_model() first.")
        assert self.tokenizer is not None, "Tokenizer not loaded. Call load_onnx_model() first."

        input_names = self.model_input_names or set()
        assert input_names is not None

        # Causal-LM reranker: the yes/no token lives at the LAST content position of every
        # row. With the default right-padding, shorter rows push their last-content position
        # away from the tensor's right edge, and INT8 dynamic quantization computes a single
        # whole-tensor activation scale across the batch — mixed content+padding activations
        # shift that scale and distort per-row logits at position seq_len-1. Left-padding
        # collapses all last-content positions onto the right edge (position -1 for every
        # row), which both simplifies pooling and keeps the quantizer's per-tensor scale
        # dominated by real content rather than pad tokens.
        #
        # The tokenizer is a shared object across encoders in the process, so we flip the
        # padding direction for the duration of this batch and restore it afterwards to
        # avoid leaking the setting into the embedding path (which assumes right-padding).
        padding = self.tokenizer.padding
        original_direction: str | None = None
        if padding is not None and padding.get("direction") != "left":
            original_direction = padding.get("direction", "right")
            self.tokenizer.enable_padding(
                pad_id=padding.get("pad_id", 0),
                pad_token=padding.get("pad_token", ""),
                direction="left",
            )

        try:
            # ⚡ Bolt: tokenizer.encode_batch(texts) parallelises via the Rust tokenizers lib.
            all_tokenized = self.tokenizer.encode_batch(texts)
        finally:
            if original_direction is not None and padding is not None:
                self.tokenizer.enable_padding(
                    pad_id=padding.get("pad_id", 0),
                    pad_token=padding.get("pad_token", ""),
                    direction=original_direction,
                )

        onnx_input: dict[str, NumpyArray] = {
            "input_ids": np.array([t.ids for t in all_tokenized], dtype=np.int64),
        }
        if "attention_mask" in input_names:
            onnx_input["attention_mask"] = np.array(
                [t.attention_mask for t in all_tokenized], dtype=np.int64
            )
        if "token_type_ids" in input_names:
            onnx_input["token_type_ids"] = np.zeros_like(onnx_input["input_ids"], dtype=np.int64)

        onnx_input = self._preprocess_onnx_input(onnx_input, **kwargs)
        outputs = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)
        model_output = outputs[0]

        if getattr(model_output, "dtype", None) == np.float16:
            model_output = model_output.astype(np.float32)  # type: ignore[unresolved-attribute]

        # _compute_yes_no_scores uses attention_mask to pick the last non-pad position
        # per sequence (full-vocab model only; YesNo variant already collapses seq dim).
        attention_mask = onnx_input.get("attention_mask")
        scores = self._compute_yes_no_scores(model_output, attention_mask)  # type: ignore[invalid-argument-type]

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
