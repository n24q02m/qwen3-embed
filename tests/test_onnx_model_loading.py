from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from qwen3_embed.common.onnx_model import OnnxModel, OnnxOutputContext
from qwen3_embed.common.types import Device


# Concrete implementation for testing
class ConcreteOnnxModel(OnnxModel[Any]):
    def _get_worker_class(cls):
        return MagicMock()

    def _post_process_onnx_output(self, output: OnnxOutputContext, **kwargs: Any):
        return []

    def load_onnx_model(self):
        pass

    def onnx_embed(self, *args: Any, **kwargs: Any) -> OnnxOutputContext:
        return OnnxOutputContext(model_output=MagicMock())


@pytest.fixture
def model():
    return ConcreteOnnxModel()


@pytest.fixture
def mock_ort():
    with patch("qwen3_embed.common.onnx_model.ort") as mock:
        # Default behavior: CPU provider available, no CUDA
        mock.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock.SessionOptions.return_value = MagicMock()
        mock.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

        # Default session mock
        session_mock = MagicMock()
        session_mock.get_providers.return_value = ["CPUExecutionProvider"]
        mock.InferenceSession.return_value = session_mock

        yield mock


def test_load_defaults(model: ConcreteOnnxModel, mock_ort):
    """Test default behavior (CPU fallback)."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    model._load_onnx_model(Path("dummy"), "model.onnx", threads=None)

    # Should use CPUExecutionProvider
    mock_ort.InferenceSession.assert_called_with(
        "dummy/model.onnx",
        providers=["CPUExecutionProvider"],
        sess_options=mock_ort.SessionOptions.return_value,
    )


def test_load_cuda_explicit(model: ConcreteOnnxModel, mock_ort):
    """Test explicit CUDA request."""
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Mock session to report CUDA provider present
    mock_ort.InferenceSession.return_value.get_providers.return_value = ["CUDAExecutionProvider"]

    model._load_onnx_model(Path("dummy"), "model.onnx", threads=None, cuda=True)

    mock_ort.InferenceSession.assert_called_with(
        "dummy/model.onnx",
        providers=["CUDAExecutionProvider"],
        sess_options=mock_ort.SessionOptions.return_value,
    )


def test_load_cuda_auto_available(model: ConcreteOnnxModel, mock_ort):
    """Test AUTO CUDA when CUDA is available."""
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Mock session to report CUDA provider present
    mock_ort.InferenceSession.return_value.get_providers.return_value = ["CUDAExecutionProvider"]

    model._load_onnx_model(Path("dummy"), "model.onnx", threads=None, cuda=Device.AUTO)

    mock_ort.InferenceSession.assert_called_with(
        "dummy/model.onnx",
        providers=["CUDAExecutionProvider"],
        sess_options=mock_ort.SessionOptions.return_value,
    )


def test_load_cuda_auto_unavailable(model: ConcreteOnnxModel, mock_ort):
    """Test AUTO CUDA when CUDA is NOT available (fallback to CPU)."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    model._load_onnx_model(Path("dummy"), "model.onnx", threads=None, cuda=Device.AUTO)

    # Should fallback to CPU
    mock_ort.InferenceSession.assert_called_with(
        "dummy/model.onnx",
        providers=["CPUExecutionProvider"],
        sess_options=mock_ort.SessionOptions.return_value,
    )


def test_load_explicit_providers(model: ConcreteOnnxModel, mock_ort):
    """Test explicit providers list."""
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Mock session to report CUDA provider present
    mock_ort.InferenceSession.return_value.get_providers.return_value = ["CUDAExecutionProvider"]

    model._load_onnx_model(
        Path("dummy"), "model.onnx", threads=None, providers=["CUDAExecutionProvider"]
    )

    mock_ort.InferenceSession.assert_called_with(
        "dummy/model.onnx",
        providers=["CUDAExecutionProvider"],
        sess_options=mock_ort.SessionOptions.return_value,
    )


def test_load_providers_validation_error(model: ConcreteOnnxModel, mock_ort):
    """Test validation error for unavailable providers."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with pytest.raises(ValueError, match="Provider CUDAExecutionProvider is not available"):
        model._load_onnx_model(
            Path("dummy"), "model.onnx", threads=None, providers=["CUDAExecutionProvider"]
        )


def test_load_cuda_and_providers_warning(model: ConcreteOnnxModel, mock_ort):
    """Test warning when both cuda and providers are specified."""
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Mock session to report CUDA provider present
    mock_ort.InferenceSession.return_value.get_providers.return_value = ["CUDAExecutionProvider"]

    with pytest.warns(UserWarning, match="`cuda` and `providers` are mutually exclusive"):
        model._load_onnx_model(
            Path("dummy"),
            "model.onnx",
            threads=None,
            cuda=True,
            providers=["CUDAExecutionProvider"],
        )


def test_load_dml_auto(model: ConcreteOnnxModel, mock_ort):
    """Test DML fallback when CUDA is not available."""
    mock_ort.get_available_providers.return_value = [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]

    model._load_onnx_model(Path("dummy"), "model.onnx", threads=None, cuda=Device.AUTO)

    mock_ort.InferenceSession.assert_called_with(
        "dummy/model.onnx",
        providers=["DmlExecutionProvider"],
        sess_options=mock_ort.SessionOptions.return_value,
    )


def test_load_threads(model: ConcreteOnnxModel, mock_ort):
    """Test thread configuration."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    model._load_onnx_model(Path("dummy"), "model.onnx", threads=4)

    so = mock_ort.SessionOptions.return_value
    assert so.intra_op_num_threads == 4
    assert so.inter_op_num_threads == 4


def test_load_extra_session_options(model: ConcreteOnnxModel, mock_ort):
    """Test extra session options."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    model._load_onnx_model(
        Path("dummy"),
        "model.onnx",
        threads=None,
        extra_session_options={"enable_cpu_mem_arena": False},
    )

    so = mock_ort.SessionOptions.return_value
    assert so.enable_cpu_mem_arena is False


def test_load_cuda_warning(model: ConcreteOnnxModel, mock_ort):
    """Test warning when CUDA is requested but session doesn't use it."""
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Simulate a scenario where we request CUDA, but the created session
    # ends up NOT using it (e.g. driver issues, although we just mock get_providers here)
    session_mock = MagicMock()
    session_mock.get_providers.return_value = ["CPUExecutionProvider"]  # CUDA missing
    mock_ort.InferenceSession.return_value = session_mock

    with pytest.warns(RuntimeWarning, match="Attempt to set CUDAExecutionProvider failed"):
        model._load_onnx_model(Path("dummy"), "model.onnx", threads=None, cuda=True)


def test_load_providers_validation_error_tuple(model: ConcreteOnnxModel, mock_ort):
    """Test validation error for unavailable tuple providers."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with pytest.raises(ValueError, match="Provider CUDAExecutionProvider is not available"):
        model._load_onnx_model(
            Path("dummy"),
            "model.onnx",
            threads=None,
            providers=[("CUDAExecutionProvider", {"device_id": 0})],
        )


def test_load_cuda_explicit_unavailable(model: ConcreteOnnxModel, mock_ort):
    """Test validation error when CUDA is explicitly requested but unavailable."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with pytest.raises(ValueError, match="Provider CUDAExecutionProvider is not available"):
        model._load_onnx_model(Path("dummy"), "model.onnx", threads=None, cuda=True)


def test_load_cuda_explicit_device_id_unavailable(model: ConcreteOnnxModel, mock_ort):
    """Test validation error when CUDA with device_id is requested but unavailable."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with pytest.raises(ValueError, match="Provider CUDAExecutionProvider is not available"):
        model._load_onnx_model(Path("dummy"), "model.onnx", threads=None, cuda=True, device_id=0)
