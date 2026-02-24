import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from qwen3_embed.common.onnx_model import OnnxModel, OnnxOutputContext
from qwen3_embed.common.types import Device


# Concrete implementation for testing abstract OnnxModel
class ConcreteOnnxModel(OnnxModel):
    def _get_worker_class(self):
        return MagicMock()

    def _post_process_onnx_output(self, output: OnnxOutputContext, **kwargs):
        return []

    def load_onnx_model(self):
        pass

    def onnx_embed(self, *args, **kwargs):
        return OnnxOutputContext(model_output=MagicMock())

class TestOnnxModelLoading(unittest.TestCase):
    def setUp(self):
        self.model = ConcreteOnnxModel()
        self.model_dir = Path("/tmp/models")
        self.model_file = "model.onnx"

        # Patches
        self.patcher_get_providers = patch("onnxruntime.get_available_providers")
        self.mock_get_providers = self.patcher_get_providers.start()

        self.patcher_session = patch("onnxruntime.InferenceSession")
        self.mock_session = self.patcher_session.start()

        self.patcher_session_options = patch("onnxruntime.SessionOptions")
        self.mock_session_options = self.patcher_session_options.start()

    def tearDown(self):
        self.patcher_get_providers.stop()
        self.patcher_session.stop()
        self.patcher_session_options.stop()

    def test_load_defaults(self):
        """Test default loading (CPU)"""
        self.mock_get_providers.return_value = ["CPUExecutionProvider"]

        self.model._load_onnx_model(
            model_dir=self.model_dir,
            model_file=self.model_file,
            threads=None
        )

        self.mock_session.assert_called_with(
            str(self.model_dir / self.model_file),
            providers=["CPUExecutionProvider"],
            sess_options=self.mock_session_options.return_value
        )

    def test_load_explicit_providers(self):
        """Test loading with explicit providers"""
        self.mock_get_providers.return_value = ["CPUExecutionProvider", "MyProvider"]

        self.model._load_onnx_model(
            model_dir=self.model_dir,
            model_file=self.model_file,
            threads=None,
            providers=["MyProvider"]
        )

        self.mock_session.assert_called_with(
            str(self.model_dir / self.model_file),
            providers=["MyProvider"],
            sess_options=self.mock_session_options.return_value
        )

    def test_load_threads(self):
        """Test setting thread options"""
        self.mock_get_providers.return_value = ["CPUExecutionProvider"]

        self.model._load_onnx_model(
            model_dir=self.model_dir,
            model_file=self.model_file,
            threads=4
        )

        so = self.mock_session_options.return_value
        self.assertEqual(so.intra_op_num_threads, 4)
        self.assertEqual(so.inter_op_num_threads, 4)

    def test_load_cuda_explicit(self):
        """Test loading with cuda=True"""
        self.mock_get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # Mock successful CUDA load
        self.mock_session.return_value.get_providers.return_value = ["CUDAExecutionProvider"]

        self.model._load_onnx_model(
            model_dir=self.model_dir,
            model_file=self.model_file,
            threads=None,
            cuda=True
        )

        self.mock_session.assert_called_with(
            str(self.model_dir / self.model_file),
            providers=["CUDAExecutionProvider"],
            sess_options=self.mock_session_options.return_value
        )

    def test_load_cuda_auto_available(self):
        """Test loading with cuda=Device.AUTO when CUDA is available"""
        self.mock_get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # Mock successful CUDA load
        self.mock_session.return_value.get_providers.return_value = ["CUDAExecutionProvider"]

        self.model._load_onnx_model(
            model_dir=self.model_dir,
            model_file=self.model_file,
            threads=None,
            cuda=Device.AUTO
        )

        self.mock_session.assert_called_with(
            str(self.model_dir / self.model_file),
            providers=["CUDAExecutionProvider"],
            sess_options=self.mock_session_options.return_value
        )

    def test_load_cuda_auto_unavailable(self):
        """Test loading with cuda=Device.AUTO when CUDA is NOT available"""
        self.mock_get_providers.return_value = ["CPUExecutionProvider"]

        self.model._load_onnx_model(
            model_dir=self.model_dir,
            model_file=self.model_file,
            threads=None,
            cuda=Device.AUTO
        )

        self.mock_session.assert_called_with(
            str(self.model_dir / self.model_file),
            providers=["CPUExecutionProvider"],
            sess_options=self.mock_session_options.return_value
        )

    def test_load_dml_auto(self):
        """Test loading with DML available and cuda=Device.AUTO"""
        self.mock_get_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]

        self.model._load_onnx_model(
            model_dir=self.model_dir,
            model_file=self.model_file,
            threads=None,
            cuda=Device.AUTO
        )

        self.mock_session.assert_called_with(
            str(self.model_dir / self.model_file),
            providers=["DmlExecutionProvider"],
            sess_options=self.mock_session_options.return_value
        )

    def test_load_providers_validation_error(self):
        """Test ValueError when invalid provider is requested"""
        self.mock_get_providers.return_value = ["CPUExecutionProvider"]

        with self.assertRaisesRegex(ValueError, "Provider InvalidProvider is not available"):
            self.model._load_onnx_model(
                model_dir=self.model_dir,
                model_file=self.model_file,
                threads=None,
                providers=["InvalidProvider"]
            )

    def test_load_cuda_and_providers_warning(self):
        """Test UserWarning when both cuda and providers are passed"""
        self.mock_get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # Mock successful CUDA load for the provider path
        self.mock_session.return_value.get_providers.return_value = ["CUDAExecutionProvider"]

        with self.assertWarns(UserWarning):
            self.model._load_onnx_model(
                model_dir=self.model_dir,
                model_file=self.model_file,
                threads=None,
                cuda=True,
                providers=["CUDAExecutionProvider"]
            )

        # It should still proceed with providers if cuda check passes/fails, but logic prioritizes providers if passed
        self.mock_session.assert_called_with(
            str(self.model_dir / self.model_file),
            providers=["CUDAExecutionProvider"],
            sess_options=self.mock_session_options.return_value
        )

    def test_load_cuda_fallback_warning(self):
        """Test RuntimeWarning when CUDA requested but session doesn't use it"""
        self.mock_get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Mock session instance to return providers without CUDA
        mock_sess_instance = self.mock_session.return_value
        mock_sess_instance.get_providers.return_value = ["CPUExecutionProvider"]

        with self.assertWarns(RuntimeWarning):
            self.model._load_onnx_model(
                model_dir=self.model_dir,
                model_file=self.model_file,
                threads=None,
                cuda=True
            )

    def test_load_device_id(self):
        """Test passing device_id with CUDA"""
        self.mock_get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # Mock successful CUDA load
        self.mock_session.return_value.get_providers.return_value = ["CUDAExecutionProvider"]

        self.model._load_onnx_model(
            model_dir=self.model_dir,
            model_file=self.model_file,
            threads=None,
            cuda=True,
            device_id=1
        )

        self.mock_session.assert_called_with(
            str(self.model_dir / self.model_file),
            providers=[("CUDAExecutionProvider", {"device_id": 1})],
            sess_options=self.mock_session_options.return_value
        )

    def test_load_extra_session_options(self):
        """Test extra session options are applied correctly"""
        self.mock_get_providers.return_value = ["CPUExecutionProvider"]

        extra_options = {"enable_cpu_mem_arena": False}

        self.model._load_onnx_model(
            model_dir=self.model_dir,
            model_file=self.model_file,
            threads=None,
            extra_session_options=extra_options
        )

        so = self.mock_session_options.return_value
        self.assertFalse(so.enable_cpu_mem_arena)

if __name__ == "__main__":
    unittest.main()
