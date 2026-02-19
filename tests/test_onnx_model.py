import onnxruntime as ort
import pytest

from qwen3_embed.common.onnx_model import OnnxModel


def test_add_extra_session_options_raises_value_error_for_unknown_option() -> None:
    so = ort.SessionOptions()
    with pytest.raises(ValueError, match="is unknown or not exposed"):
        OnnxModel.add_extra_session_options(so, {"invalid_option": True})


def test_add_extra_session_options_sets_valid_option() -> None:
    so = ort.SessionOptions()
    assert so.enable_cpu_mem_arena is True
    OnnxModel.add_extra_session_options(so, {"enable_cpu_mem_arena": False})
    assert so.enable_cpu_mem_arena is False
