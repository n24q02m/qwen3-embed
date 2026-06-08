"""Extra tests for GGUF Cross Encoder scoring edge cases."""

from unittest.mock import MagicMock

import numpy as np

from qwen3_embed.rerank.cross_encoder.gguf_cross_encoder import TOKEN_NO_ID, TOKEN_YES_ID
from tests.test_gguf_cross_encoder import _make_model


class TestScoreTextExtra:
    def test_score_text_inf_diff(self):
        """Test _score_text handles infinite positive diff."""
        model = _make_model()
        logits = np.zeros(10000, dtype=np.float32)
        logits[TOKEN_YES_ID] = float("inf")
        logits[TOKEN_NO_ID] = 0.0
        model._llm.scores.__getitem__ = MagicMock(return_value=logits)

        score = model._score_text("some text")
        assert score == 1.0

    def test_score_text_neg_inf_diff(self):
        """Test _score_text handles infinite negative diff."""
        model = _make_model()
        logits = np.zeros(10000, dtype=np.float32)
        logits[TOKEN_YES_ID] = 0.0
        logits[TOKEN_NO_ID] = float("inf")
        model._llm.scores.__getitem__ = MagicMock(return_value=logits)

        score = model._score_text("some text")
        assert score == 0.0
