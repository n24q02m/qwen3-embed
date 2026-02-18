"""Unit tests for Qwen3CrossEncoder model registration and scoring logic."""

import numpy as np

from qwen3_embed.rerank.cross_encoder.qwen3_cross_encoder import (
    DEFAULT_INSTRUCTION,
    SYSTEM_PROMPT,
    TOKEN_NO_ID,
    TOKEN_YES_ID,
    Qwen3CrossEncoder,
    supported_qwen3_reranker_models,
)
from qwen3_embed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder


class TestQwen3CrossEncoderRegistry:
    """Verify Qwen3 reranker models are properly registered."""

    def test_qwen3_cross_encoder_is_in_registry(self):
        """Qwen3CrossEncoder should be in the TextCrossEncoder registry."""
        assert Qwen3CrossEncoder in TextCrossEncoder.CROSS_ENCODER_REGISTRY

    def test_list_supported_models_contains_qwen3_reranker(self):
        """Qwen3 reranker models should appear in list_supported_models."""
        models = TextCrossEncoder.list_supported_models()
        qwen3_models = [m for m in models if "Qwen3" in m["model"]]
        assert len(qwen3_models) >= 1

    def test_supported_qwen3_reranker_models_has_correct_description(self):
        """Verify Qwen3-Reranker-0.6B model description fields."""
        desc = supported_qwen3_reranker_models[0]
        assert desc.model == "n24q02m/Qwen3-Reranker-0.6B-ONNX"
        assert desc.license == "apache-2.0"
        assert "yes/no" in desc.description.lower() or "causal" in desc.description.lower()


class TestQwen3ChatTemplate:
    """Verify chat template formatting for reranking."""

    def test_format_rerank_input_returns_correct_template(self):
        result = Qwen3CrossEncoder._format_rerank_input(
            query="What is AI?",
            document="AI is artificial intelligence.",
            instruction=DEFAULT_INSTRUCTION,
        )
        assert "<|im_start|>system" in result
        assert SYSTEM_PROMPT in result
        assert "<Instruct>:" in result
        assert "<Query>: What is AI?" in result
        assert "<Document>: AI is artificial intelligence." in result
        assert "<|im_end|>" in result
        assert "<think>" in result
        assert "</think>" in result

    def test_format_rerank_input_uses_custom_instruction(self):
        result = Qwen3CrossEncoder._format_rerank_input(
            query="q",
            document="d",
            instruction="Custom task instruction",
        )
        assert "<Instruct>: Custom task instruction" in result


class TestYesNoScoring:
    """Test the yes/no softmax scoring logic."""

    def test_compute_yes_no_scores_strong_yes_returns_high_score(self):
        """When yes-logit >> no-logit, score should be close to 1.0."""
        # (batch=1, seq_len=3, vocab_size=20000)
        vocab_size = 20000
        output = np.zeros((1, 3, vocab_size), dtype=np.float32)
        output[0, -1, TOKEN_YES_ID] = 10.0
        output[0, -1, TOKEN_NO_ID] = -10.0

        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        assert scores.shape == (1,)
        assert scores[0] > 0.99

    def test_compute_yes_no_scores_strong_no_returns_low_score(self):
        """When no-logit >> yes-logit, score should be close to 0.0."""
        vocab_size = 20000
        output = np.zeros((1, 3, vocab_size), dtype=np.float32)
        output[0, -1, TOKEN_YES_ID] = -10.0
        output[0, -1, TOKEN_NO_ID] = 10.0

        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        assert scores[0] < 0.01

    def test_compute_yes_no_scores_equal_logits_returns_half(self):
        """When yes==no logits, score should be 0.5."""
        vocab_size = 20000
        output = np.zeros((1, 3, vocab_size), dtype=np.float32)
        output[0, -1, TOKEN_YES_ID] = 5.0
        output[0, -1, TOKEN_NO_ID] = 5.0

        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        np.testing.assert_allclose(scores[0], 0.5, atol=1e-6)

    def test_compute_yes_no_scores_batch_returns_correct_scores(self):
        """Batch of 3 samples with varying relevance."""
        vocab_size = 20000
        output = np.zeros((3, 5, vocab_size), dtype=np.float32)

        # Sample 0: strong yes
        output[0, -1, TOKEN_YES_ID] = 10.0
        output[0, -1, TOKEN_NO_ID] = -10.0

        # Sample 1: strong no
        output[1, -1, TOKEN_YES_ID] = -10.0
        output[1, -1, TOKEN_NO_ID] = 10.0

        # Sample 2: neutral
        output[2, -1, TOKEN_YES_ID] = 0.0
        output[2, -1, TOKEN_NO_ID] = 0.0

        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        assert scores.shape == (3,)
        assert scores[0] > 0.99
        assert scores[1] < 0.01
        np.testing.assert_allclose(scores[2], 0.5, atol=1e-6)

    def test_compute_yes_no_scores_large_logits_handles_stability(self):
        """Large logit values should not cause overflow."""
        vocab_size = 20000
        output = np.zeros((1, 3, vocab_size), dtype=np.float32)
        output[0, -1, TOKEN_YES_ID] = 1000.0
        output[0, -1, TOKEN_NO_ID] = 999.0

        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        assert not np.isnan(scores[0])
        assert not np.isinf(scores[0])
        assert 0.5 < scores[0] < 1.0


class TestTokenConstants:
    """Verify token ID constants."""

    def test_token_ids_are_positive(self):
        assert TOKEN_YES_ID > 0
        assert TOKEN_NO_ID > 0

    def test_token_ids_are_different(self):
        assert TOKEN_YES_ID != TOKEN_NO_ID
