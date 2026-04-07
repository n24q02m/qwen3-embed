from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qwen3_embed.common.onnx_model import EmbeddingWorker, OnnxModel, OnnxOutputContext
from qwen3_embed.common.types import Device


# Concrete implementation for testing OnnxModel
class ConcreteOnnxModel(OnnxModel[Any]):
    @classmethod
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

    with pytest.warns(UserWarning, match=r"`cuda` and `providers` are mutually exclusive"):
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


def test_add_extra_session_options():
    session_options = MagicMock()

    # Test valid options
    ConcreteOnnxModel.add_extra_session_options(session_options, {"enable_cpu_mem_arena": False})
    assert session_options.enable_cpu_mem_arena is False

    # Test valid options (True)
    ConcreteOnnxModel.add_extra_session_options(session_options, {"enable_cpu_mem_arena": True})
    assert session_options.enable_cpu_mem_arena is True

    # Test invalid option
    with pytest.raises(
        ValueError,
        match=r"invalid_option is unknown or not exposed \(exposed options: \('enable_cpu_mem_arena',\)\)",
    ):
        ConcreteOnnxModel.add_extra_session_options(session_options, {"invalid_option": True})


# Concrete implementation for testing EmbeddingWorker
class ConcreteEmbeddingWorker(EmbeddingWorker[Any]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> OnnxModel[Any]:
        return ConcreteOnnxModel()

    def process(self, items):
        return []


def test_embedding_worker_init():
    worker = ConcreteEmbeddingWorker(model_name="test_model", cache_dir="test_cache")
    assert isinstance(worker.model, ConcreteOnnxModel)


def test_embedding_worker_start():
    worker = ConcreteEmbeddingWorker.start(model_name="test_model", cache_dir="test_cache")
    assert isinstance(worker, ConcreteEmbeddingWorker)
    assert isinstance(worker.model, ConcreteOnnxModel)


def test_onnx_model_abstract_methods():
    model = OnnxModel()
    with pytest.raises(NotImplementedError):
        model._get_worker_class()
    with pytest.raises(NotImplementedError):
        model._post_process_onnx_output(OnnxOutputContext(model_output=np.array([])))
    with pytest.raises(NotImplementedError):
        model.load_onnx_model()
    with pytest.raises(NotImplementedError):
        model.onnx_embed()


def test_embedding_worker_abstract_methods():
    class IncompleteWorker(EmbeddingWorker[Any]):
        pass

    with pytest.raises(NotImplementedError):
        IncompleteWorker(model_name="test", cache_dir="test")

    worker = ConcreteEmbeddingWorker(model_name="test", cache_dir="test")
    with pytest.raises(NotImplementedError):
        super(ConcreteEmbeddingWorker, worker).process([])


def test_preprocess_onnx_input(model):
    input_data = {"input": np.array([1, 2, 3])}
    processed = model._preprocess_onnx_input(input_data)
    assert processed == input_data


def test_select_exposed_session_options():
    kwargs = {"enable_cpu_mem_arena": True, "other_param": 123}
    selected = ConcreteOnnxModel._select_exposed_session_options(kwargs)
    assert selected == {"enable_cpu_mem_arena": True}


def test_onnx_output_context_instantiation():
    output = np.array([0.1, 0.2])
    mask = np.array([1, 1], dtype=np.int64)
    ids = np.array([10, 20], dtype=np.int64)
    metadata = {"key": "value"}

    context = OnnxOutputContext(
        model_output=output, attention_mask=mask, input_ids=ids, metadata=metadata
    )

    assert np.array_equal(context.model_output, output)
    assert context.attention_mask is not None
    assert np.array_equal(context.attention_mask, mask)
    assert context.input_ids is not None
    assert np.array_equal(context.input_ids, ids)
    assert context.metadata == metadata


def test_onnx_output_context_defaults():
    output = np.array([0.1, 0.2])
    context = OnnxOutputContext(model_output=output)

    assert np.array_equal(context.model_output, output)
    assert context.attention_mask is None
    assert context.input_ids is None
    assert context.metadata is None
