"""Tests for the optional HF-id to ONNX export helper."""

import importlib.util
import sys
from unittest.mock import patch

import pytest

from qwen3_embed.export import export_to_onnx

_HAS_OPTIMUM = importlib.util.find_spec("optimum") is not None


@pytest.mark.skipif(_HAS_OPTIMUM, reason="exercises the missing-extra path; optimum is installed")
def test_export_without_extra_raises_helpful_error(tmp_path):
    with pytest.raises(ImportError, match=r"optimum\[exporters\]"):
        export_to_onnx("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path))


def test_export_missing_dependency_raises_error(tmp_path):
    """Test that export_to_onnx raises ImportError when optimum is missing,
    even if it's installed in the environment (using mocks).
    """
    with patch.dict(sys.modules, {"optimum.exporters.onnx": None}):
        # We need to reload or ensure the import is attempted inside the patch
        # Since export_to_onnx does a lazy import inside the function,
        # patching sys.modules should work.
        with pytest.raises(ImportError, match=r"optimum\[exporters\]"):
            export_to_onnx("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path))


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_OPTIMUM, reason="requires the [export] extra")
def test_export_minilm_roundtrip(tmp_path):
    import onnxruntime as ort  # noqa: F401

    out = export_to_onnx("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path))
    assert (tmp_path / "model.onnx").exists() or (tmp_path / "onnx" / "model.onnx").exists()
    assert out == str(tmp_path)
