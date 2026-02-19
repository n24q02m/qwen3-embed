import pytest

from qwen3_embed.text.onnx_text_model import OnnxTextModel


class ConcreteOnnxTextModel(OnnxTextModel):
    def load_onnx_model(self):
        pass

    @classmethod
    def _get_worker_class(cls):
        pass

    def _post_process_onnx_output(self, output, **kwargs):
        pass


def test_onnx_embed_raises_when_uninitialized():
    model = ConcreteOnnxTextModel()
    # Explicitly do NOT load the model (model.load_onnx_model() is not called)

    with pytest.raises(ValueError, match="Model is not loaded"):
        model.onnx_embed(["test"])
