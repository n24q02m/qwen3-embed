"""Unit tests for Qwen3TextEmbedding model registration, configuration, and post-processing."""

import inspect

import numpy as np
import pytest

from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.text.qwen3_embedding import (
    DEFAULT_TASK,
    QUERY_INSTRUCTION_TEMPLATE,
    Qwen3TextEmbedding,
    supported_qwen3_models,
)
from qwen3_embed.text.text_embedding import TextEmbedding


class TestQwen3TextEmbeddingRegistry:
    """Verify Qwen3 embedding models are properly registered."""

    def test_registry_when_initialized_contains_qwen3(self):
        """Qwen3TextEmbedding should be in the TextEmbedding registry."""
        assert Qwen3TextEmbedding in TextEmbedding.EMBEDDINGS_REGISTRY

    def test_list_supported_models_when_called_contains_qwen3_models(self):
        """Qwen3 models should appear in list_supported_models."""
        models = TextEmbedding.list_supported_models()
        qwen3_models = [m for m in models if "Qwen3" in m["model"]]
        assert len(qwen3_models) >= 1

    def test_model_description_when_checked_has_correct_fields(self):
        """Verify Qwen3-Embedding-0.6B model description fields."""
        desc = supported_qwen3_models[0]
        assert desc.model == "Qwen/Qwen3-Embedding-0.6B"
        assert desc.dim == 1024
        assert desc.license == "apache-2.0"
        assert "last-token" in desc.description.lower() or "MRL" in desc.description

    def test_init_when_model_name_default_is_qwen3(self):
        """TextEmbedding default model should be Qwen3."""
        sig = inspect.signature(TextEmbedding.__init__)
        default_model = sig.parameters["model_name"].default
        assert default_model == "Qwen/Qwen3-Embedding-0.6B"


class TestQwen3InstructionFormat:
    """Verify instruction template formatting."""

    def test_format_when_query_instruction_template_used_returns_correct_string(self):
        result = QUERY_INSTRUCTION_TEMPLATE.format(task="Find docs", text="test query")
        assert result == "Instruct: Find docs\nQuery: test query"

    def test_default_task_when_checked_is_not_empty(self):
        assert len(DEFAULT_TASK) > 0
        assert "retrieve" in DEFAULT_TASK.lower()


class TestQwen3PostProcessing:
    """Test _post_process_onnx_output with mocked ONNX output."""

    def test_post_process_when_last_token_pooling_returns_normalized_embedding(self):
        """Should extract last-token embeddings and normalise."""
        model_output = np.array(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.0, 0.0, 0.0, 0.0]],
                [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6], [1.7, 1.8, 1.9, 2.0]],
            ]
        )
        attention_mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)
        output = OnnxOutputContext(model_output=model_output, attention_mask=attention_mask)

        result = np.array(
            list(Qwen3TextEmbedding._post_process_onnx_output(None, output))  # type: ignore[arg-type]
        )

        assert result.shape == (2, 4)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_post_process_when_dim_specified_returns_truncated_embedding(self):
        """Passing dim= should truncate before normalisation."""
        model_output = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
        attention_mask = np.array([[1, 1]], dtype=np.int64)
        output = OnnxOutputContext(model_output=model_output, attention_mask=attention_mask)

        result = np.array(
            list(Qwen3TextEmbedding._post_process_onnx_output(None, output, dim=2))  # type: ignore[arg-type]
        )
        assert result.shape == (1, 2)
        np.testing.assert_allclose(np.linalg.norm(result[0]), 1.0, atol=1e-6)

    def test_post_process_when_attention_mask_missing_raises_value_error(self):
        """Should raise if attention_mask is None."""
        output = OnnxOutputContext(model_output=np.zeros((1, 3, 4)), attention_mask=None)
        with pytest.raises(ValueError, match="attention_mask"):
            list(Qwen3TextEmbedding._post_process_onnx_output(None, output))  # type: ignore[arg-type]
