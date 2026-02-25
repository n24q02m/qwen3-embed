from unittest.mock import MagicMock, patch

import pytest

from qwen3_embed.text.onnx_text_model import OnnxTextModel


class DummyTextModel(OnnxTextModel):
    def load_onnx_model(self):
        pass

    @classmethod
    def _get_worker_class(cls):
        pass

    def _post_process_onnx_output(self, output, **kwargs):
        return []


@pytest.fixture
def mock_onnx_session():
    with patch("onnxruntime.InferenceSession") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        # Mock get_inputs to return a list of nodes with names
        input_node = MagicMock()
        input_node.name = "input_ids"
        mock_session.get_inputs.return_value = [input_node]
        mock_session.run.return_value = [MagicMock()]
        yield mock_session


@pytest.fixture
def mock_tokenizer():
    with patch("qwen3_embed.text.onnx_text_model.load_tokenizer") as mock_load:
        tokenizer = MagicMock()
        # Mock encoding object
        encoding = MagicMock()
        encoding.ids = [1, 2]
        encoding.attention_mask = [1, 1]
        encoding.type_ids = [0, 0]
        tokenizer.encode_batch.return_value = [encoding]

        mock_load.return_value = (tokenizer, {})
        yield tokenizer


def test_onnx_embed_calls_get_inputs_only_once(mock_onnx_session, mock_tokenizer):
    model = DummyTextModel()

    # Mock filesystem interactions for _load_onnx_model
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", new_callable=MagicMock),
    ):
        # Load the model
        model._load_onnx_model(model_dir=MagicMock(), model_file="model.onnx", threads=1)

    # get_inputs might be called during loading (e.g. for inspection, though current code doesn't seem to)
    # Let's check the count after loading.
    load_call_count = mock_onnx_session.get_inputs.call_count

    # Run embed
    model.onnx_embed(["test"])

    # Assert that get_inputs call count has NOT increased
    # Before the fix, this assertion should FAIL because onnx_embed calls get_inputs
    assert mock_onnx_session.get_inputs.call_count == load_call_count, (
        f"get_inputs called {mock_onnx_session.get_inputs.call_count - load_call_count} times during embed"
    )
