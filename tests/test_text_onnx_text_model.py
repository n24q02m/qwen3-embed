"""Tests for OnnxTextModel and OnnxTextEmbedding — covers onnx_text_model.py and onnx_embedding.py."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.common.onnx_model import OnnxInferenceConfig, OnnxOutputContext
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.text.onnx_embedding import (
    OnnxTextEmbedding,
    OnnxTextEmbeddingWorker,
    supported_onnx_models,
)
from qwen3_embed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "test-org/test-onnx-model"
_MODEL_DESC = DenseModelDescription(
    model=_MODEL_NAME,
    sources=ModelSource(hf="test-org/test-onnx-model"),
    model_file="onnx/model.onnx",
    description="test",
    license="MIT",
    size_in_GB=0.1,
    dim=4,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tokenizer(n_docs: int = 1, seq_len: int = 4) -> MagicMock:
    ids = [1, 2, 3, 0][:seq_len]
    mask = [1, 1, 1, 0][:seq_len]
    enc = MagicMock()
    enc.ids = ids
    enc.attention_mask = mask
    tok = MagicMock()
    tok.encode_batch.return_value = [enc] * n_docs
    return tok


def _make_mock_session(output_shape: tuple[int, ...] = (1, 4), dtype=np.float32) -> MagicMock:
    output = np.ones(output_shape, dtype=dtype)
    session = MagicMock()
    session.run.return_value = [output]
    node_ids = MagicMock()
    node_ids.name = "input_ids"
    node_attn = MagicMock()
    node_attn.name = "attention_mask"
    session.get_inputs.return_value = [node_ids, node_attn]
    return session


@pytest.fixture()
def registered_test_model() -> Iterable[DenseModelDescription]:
    """Add the test model to OnnxTextEmbedding.supported_onnx_models."""
    supported_onnx_models.append(_MODEL_DESC)
    yield _MODEL_DESC
    if _MODEL_DESC in supported_onnx_models:
        supported_onnx_models.remove(_MODEL_DESC)


# ---------------------------------------------------------------------------
# Concrete OnnxTextModel subclass for unit-testing the base class
# ---------------------------------------------------------------------------


class ConcreteOnnxTextModel(OnnxTextModel[NumpyArray]):
    """Minimal concrete subclass — does not load any real models."""

    @classmethod
    def _get_worker_class(cls) -> type["TextEmbeddingWorker[NumpyArray]"]:
        return _StubWorker

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        yield output.model_output

    def load_onnx_model(self) -> None:
        pass  # no-op in tests


class _StubWorker(TextEmbeddingWorker[NumpyArray]):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextModel[NumpyArray]:
        m = ConcreteOnnxTextModel()
        m.model = _make_mock_session()
        m.tokenizer = _make_mock_tokenizer()
        return m


# ---------------------------------------------------------------------------
# OnnxTextModel — covers onnx_text_model.py
# ---------------------------------------------------------------------------


class TestOnnxTextModelMethods:
    """Lines 31-118 — basic methods."""

    def test_init_sets_defaults(self) -> None:
        m = ConcreteOnnxTextModel()
        assert m.tokenizer is None
        assert m.special_token_to_id == {}

    def test_preprocess_returns_unchanged(self) -> None:
        m = ConcreteOnnxTextModel()
        data: dict[str, NumpyArray] = {"input_ids": np.array([[1, 2, 3]])}
        assert m._preprocess_onnx_input(data) is data

    def test_load_onnx_model_abstract_raises(self) -> None:
        m = OnnxTextModel()
        with pytest.raises(NotImplementedError):
            m.load_onnx_model()

    def test_onnx_embed_requires_model_loaded(self) -> None:
        m = ConcreteOnnxTextModel()
        m.model = None
        with pytest.raises(ValueError, match="Model not loaded"):
            m.onnx_embed(["hello"])

    def test_onnx_embed_runs_session(self) -> None:
        m = ConcreteOnnxTextModel()
        m.model = _make_mock_session()
        m.model_input_names = {"input_ids", "attention_mask"}
        m.tokenizer = _make_mock_tokenizer()

        ctx = m.onnx_embed(["hello"])
        assert ctx.model_output.shape == (1, 4)
        assert ctx.attention_mask is not None
        assert ctx.input_ids is not None


class TestEmbedDocumentsBranching:
    """Lines 124-165 — _embed_documents in single-process and parallel modes."""

    def _loaded(self, n_docs: int = 1) -> ConcreteOnnxTextModel:
        m = ConcreteOnnxTextModel()
        m.model = _make_mock_session(output_shape=(n_docs, 4))
        m.model_input_names = {"input_ids", "attention_mask"}
        m.tokenizer = _make_mock_tokenizer(n_docs=n_docs)
        return m

    def _parallel_docs(self) -> list[str]:
        return ["a", "b", "c", "d"]

    def test_single_string_wrapped_in_list(self) -> None:
        m = self._loaded()
        config = OnnxInferenceConfig(model_name="t", cache_dir="/tmp")
        results = list(m._embed_documents(documents="hello", config=config))
        assert len(results) == 1

    def test_list_smaller_than_batch_size_skips_parallel(self) -> None:
        m = self._loaded(n_docs=2)
        config = OnnxInferenceConfig(model_name="t", cache_dir="/tmp")
        # documents < batch_size => parallel branch not taken
        results = list(m._embed_documents(documents=["a", "b"], config=config, batch_size=64))
        assert len(results) == 1  # yield output.model_output as a whole in ConcreteOnnxTextModel
        assert results[0].shape == (2, 4)

    def test_parallel_none_uses_direct_path(self) -> None:
        m = self._loaded()
        config = OnnxInferenceConfig(model_name="t", cache_dir="/tmp")
        with patch("qwen3_embed.text.onnx_text_model.ParallelWorkerPool") as mock_cls:
            list(m._embed_documents(documents=["hi"], config=config, parallel=None))
        mock_cls.assert_not_called()

    def test_parallel_zero_uses_cpu_count(self) -> None:
        m = self._loaded()
        config = OnnxInferenceConfig(model_name="t", cache_dir="/tmp")
        out = OnnxOutputContext(model_output=np.ones((1, 4), dtype=np.float32))
        with patch("qwen3_embed.text.onnx_text_model.ParallelWorkerPool") as mock_cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            mock_cls.return_value = pool
            list(
                m._embed_documents(
                    documents=self._parallel_docs(),
                    config=config,
                    batch_size=2,
                    parallel=0,
                )
            )
        mock_cls.assert_called_once()

    def test_parallel_positive_sets_num_workers(self) -> None:
        m = self._loaded()
        config = OnnxInferenceConfig(model_name="t", cache_dir="/tmp")
        out = OnnxOutputContext(model_output=np.ones((1, 4), dtype=np.float32))
        with patch("qwen3_embed.text.onnx_text_model.ParallelWorkerPool") as mock_cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            mock_cls.return_value = pool
            list(
                m._embed_documents(
                    documents=self._parallel_docs(),
                    config=config,
                    batch_size=2,
                    parallel=3,
                )
            )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["num_workers"] == 3

    def test_parallel_extra_session_options_merged(self) -> None:
        m = self._loaded()
        config = OnnxInferenceConfig(
            model_name="t",
            cache_dir="/tmp",
            extra_session_options={"enable_cpu_mem_arena": False},
        )
        out = OnnxOutputContext(model_output=np.ones((1, 4), dtype=np.float32))
        with patch("qwen3_embed.text.onnx_text_model.ParallelWorkerPool") as mock_cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            mock_cls.return_value = pool
            list(
                m._embed_documents(
                    documents=self._parallel_docs(),
                    config=config,
                    batch_size=2,
                    parallel=2,
                )
            )
        mock_cls.assert_called_once()
        # verify options merged into params passed to ordered_map
        _, call_kwargs = pool.ordered_map.call_args
        assert call_kwargs["enable_cpu_mem_arena"] is False

    def test_parallel_uses_forkserver_or_spawn(self) -> None:
        """start_method selection doesn't crash."""
        m = self._loaded()
        config = OnnxInferenceConfig(model_name="t", cache_dir="/tmp")
        out = OnnxOutputContext(model_output=np.ones((1, 4), dtype=np.float32))
        with patch("qwen3_embed.text.onnx_text_model.ParallelWorkerPool") as mock_cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            mock_cls.return_value = pool
            list(
                m._embed_documents(
                    documents=self._parallel_docs(),
                    config=config,
                    batch_size=2,
                    parallel=1,
                )
            )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["start_method"] in ("forkserver", "spawn")


class TestTextEmbeddingWorkerProcess:
    """Lines 183-185 — TextEmbeddingWorker.process yields (idx, OnnxOutputContext)."""

    def test_yields_indexed_outputs(self) -> None:
        worker = _StubWorker(model_name="t", cache_dir="/tmp")
        items = [(0, ["hello"]), (1, ["world"])]
        results = list(worker.process(items))
        assert len(results) == 2
        assert results[0][0] == 0
        assert isinstance(results[0][1], OnnxOutputContext)
        assert results[1][0] == 1


# ===========================================================================
# OnnxTextEmbedding — covers onnx_embedding.py
# ===========================================================================


class TestOnnxTextEmbeddingInit:
    """Lines 65-91 — OnnxTextEmbedding.__init__ behaviour."""

    def test_lazy_load_skips_session(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
            emb = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)
        assert emb.lazy_load is True
        # Due to MRO: TextEmbeddingBase.__init__ does not call super().__init__(),
        # so OnnxModel.__init__ is never invoked and self.model is never set.
        assert not hasattr(emb, "model") or emb.model is None

    def test_lazy_load_false_calls_load_onnx_model(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        called: list[bool] = []
        with (
            patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path),
            patch.object(
                OnnxTextEmbedding, "load_onnx_model", side_effect=lambda: called.append(True)
            ),
        ):
            OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=False)
        assert called

    def test_device_id_set_directly(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
            emb = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True, device_id=7)
        assert emb.device_id == 7

    def test_device_id_from_device_ids(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
            emb = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True, device_ids=[2, 3])
        assert emb.device_id == 2

    def test_no_device_id_defaults_to_none(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
            emb = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)
        assert emb.device_id is None

    def test_specific_model_path_stored(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
            emb = OnnxTextEmbedding(
                model_name=_MODEL_NAME,
                lazy_load=True,
                specific_model_path="/custom",
            )
        assert emb._specific_model_path == "/custom"


@pytest.fixture()
def onnx_emb(tmp_path: Path, registered_test_model: DenseModelDescription) -> OnnxTextEmbedding:
    """OnnxTextEmbedding with lazy_load + pre-wired mock session and tokenizer."""
    with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
        emb = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)

    session = _make_mock_session((1, 4))
    tok = _make_mock_tokenizer(n_docs=1)
    emb.model = session
    emb.model_input_names = {"input_ids", "attention_mask"}
    emb.tokenizer = tok
    return emb


class TestOnnxTextEmbeddingMethods:
    def test_preprocess_returns_unchanged(self, onnx_emb: OnnxTextEmbedding) -> None:
        """Line 140."""
        data: dict[str, NumpyArray] = {"input_ids": np.array([[1, 2, 3]])}
        assert onnx_emb._preprocess_onnx_input(data) is data

    def test_get_worker_class(self, onnx_emb: OnnxTextEmbedding) -> None:
        """Line 132."""
        assert onnx_emb._get_worker_class() is OnnxTextEmbeddingWorker

    def test_post_process_2d_output(self, onnx_emb: OnnxTextEmbedding) -> None:
        """Line 150, 153 — 2D embeddings are normalized row-wise."""
        output = OnnxOutputContext(model_output=np.ones((2, 4), dtype=np.float32))
        results = list(onnx_emb._post_process_onnx_output(output))
        assert len(results) == 2
        assert results[0].shape == (4,)

    def test_post_process_3d_output_takes_cls(self, onnx_emb: OnnxTextEmbedding) -> None:
        """Line 148, 153 — 3D embeddings use CLS (first) token."""
        output = OnnxOutputContext(model_output=np.ones((2, 5, 4), dtype=np.float32))
        results = list(onnx_emb._post_process_onnx_output(output))
        assert len(results) == 2
        assert results[0].shape == (4,)

    def test_post_process_unsupported_ndim_raises(self, onnx_emb: OnnxTextEmbedding) -> None:
        output = OnnxOutputContext(model_output=np.ones((1, 2, 3, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="Unsupported embedding shape"):
            list(onnx_emb._post_process_onnx_output(output))

    def test_embed_yields_normalized_embeddings(self, onnx_emb: OnnxTextEmbedding) -> None:
        """Line 115."""
        results = list(onnx_emb.embed(["hello"]))
        assert len(results) == 1
        assert results[0].shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(results[0]), 1.0, atol=1e-6)

    def test_token_count_sums_mask(self, onnx_emb: OnnxTextEmbedding) -> None:
        """Line 169."""
        enc = onnx_emb.tokenizer.encode_batch.return_value[0]  # type: ignore[unresolved-attribute]
        enc.attention_mask = [1, 1, 0, 0]
        assert onnx_emb.token_count("hello") == 2

    def test_load_onnx_model_delegates_to_load_onnx_model(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        """Line 156 — load_onnx_model calls _load_onnx_model with correct args."""
        with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
            emb = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)

        with patch.object(emb, "_load_onnx_model") as mock_load:
            emb.load_onnx_model()

        mock_load.assert_called_once_with(
            model_dir=tmp_path,
            model_file=_MODEL_DESC.model_file,
            threads=emb.threads,
            providers=emb.providers,
            cuda=emb.cuda,
            device_id=emb.device_id,
            extra_session_options=emb._extra_session_options,
        )


class TestOnnxTextEmbeddingWorkerInit:
    """Line 179 — OnnxTextEmbeddingWorker.init_embedding creates OnnxTextEmbedding."""

    def test_creates_onnx_text_embedding(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        with (
            patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path),
            patch.object(OnnxTextEmbedding, "load_onnx_model"),
        ):
            worker = OnnxTextEmbeddingWorker(
                model_name=_MODEL_NAME,
                cache_dir=str(tmp_path),
            )
        assert isinstance(worker.model, OnnxTextEmbedding)
        assert worker.model.model_name == _MODEL_NAME
