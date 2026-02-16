from unittest.mock import MagicMock

import numpy as np

from qwen3_embed.text.onnx_text_model import OnnxTextModel


class MockOnnxTextModel(OnnxTextModel):
    """Subclass to test onnx_embed without a real model."""

    def __init__(self):
        # Initialize parent but don't call load_onnx_model
        # We manually set what we need
        self.tokenizer = MagicMock()
        self.model = MagicMock()
        self.ONNX_OUTPUT_NAMES = ["output"]
        self.special_token_to_id = {}

    # We need to implement abstract methods
    @classmethod
    def _get_worker_class(cls):
        pass

    def _post_process_onnx_output(self, output, **kwargs):
        pass

    def load_onnx_model(self):
        pass


def test_token_type_ids_optimization():
    model = MockOnnxTextModel()

    # Mock tokenizer output
    mock_encoding = MagicMock()
    mock_encoding.ids = [1, 2, 3, 4, 5]
    mock_encoding.attention_mask = [1, 1, 1, 1, 1]
    # Simulate batch of 2
    model.tokenize = MagicMock(return_value=[mock_encoding, mock_encoding])

    # Mock model inputs to include token_type_ids and attention_mask
    mock_input_node_ids = MagicMock()
    mock_input_node_ids.name = "input_ids"

    mock_input_node_type = MagicMock()
    mock_input_node_type.name = "token_type_ids"

    mock_input_node_att = MagicMock()
    mock_input_node_att.name = "attention_mask"

    model.model.get_inputs.return_value = [
        mock_input_node_ids,
        mock_input_node_type,
        mock_input_node_att,
    ]

    # Mock model run output
    model.model.run.return_value = [np.array([[0.1], [0.2]], dtype=np.float32)]

    # Run onnx_embed
    documents = ["doc1", "doc2"]
    model.onnx_embed(documents)

    # Verify model.run was called with optimized token_type_ids
    args, _ = model.model.run.call_args
    output_names, onnx_input = args

    assert "token_type_ids" in onnx_input
    token_type_ids = onnx_input["token_type_ids"]

    # Check shape and values
    assert token_type_ids.shape == (2, 5)
    assert token_type_ids.dtype == np.int64
    np.testing.assert_array_equal(token_type_ids, np.zeros((2, 5), dtype=np.int64))

    # Check if input_ids is correct
    assert "input_ids" in onnx_input
    assert onnx_input["input_ids"].shape == (2, 5)
    np.testing.assert_array_equal(
        onnx_input["input_ids"], np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=np.int64)
    )
