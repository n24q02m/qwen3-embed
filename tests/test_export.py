"""Tests for the optional HF-id to ONNX export helper."""

import importlib.util

import pytest

from qwen3_embed.export import export_to_onnx

_HAS_OPTIMUM = importlib.util.find_spec("optimum") is not None


@pytest.mark.skipif(
    _HAS_OPTIMUM, reason="exercises the missing-extra path; optimum is installed"
)
def test_export_without_extra_raises_helpful_error(tmp_path):
    with pytest.raises(ImportError, match=r"optimum\[exporters\]"):
        export_to_onnx("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path))


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_OPTIMUM, reason="requires the [export] extra")
def test_export_minilm_roundtrip(tmp_path):
    import onnxruntime as ort  # noqa: F401

    out = export_to_onnx("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path))
    assert (tmp_path / "model.onnx").exists() or (tmp_path / "onnx" / "model.onnx").exists()
    assert out == str(tmp_path)
