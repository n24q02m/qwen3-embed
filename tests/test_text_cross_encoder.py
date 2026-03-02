"""Tests for TextCrossEncoder functionality."""

import pytest

from qwen3_embed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder


def test_init_unsupported_model_raises_value_error():
    """Initializing TextCrossEncoder with an unsupported model should raise ValueError."""
    with pytest.raises(
        ValueError,
        match="Model unsupported-model-name is not supported in TextCrossEncoder. Please check the supported models using `TextCrossEncoder.list_supported_models\\(\\)`",
    ):
        TextCrossEncoder(model_name="unsupported-model-name")
