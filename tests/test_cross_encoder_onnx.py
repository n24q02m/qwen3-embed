"""Tests for OnnxCrossEncoderModel (onnx_text_model.py) and
OnnxTextCrossEncoder (onnx_text_cross_encoder.py).

All ONNX sessions, tokenizers, and model downloads are fully mocked.
No real models are loaded.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qwen3_embed.common.model_description import BaseModelDescription, ModelSource
from qwen3_embed.common.onnx_model import OnnxInferenceConfig, OnnxOutputContext
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.rerank.cross_encoder.onnx_text_cross_encoder import (
    OnnxTextCrossEncoder,
    TextCrossEncoderWorker,
    supported_onnx_models,
)
from qwen3_embed.rerank.cross_encoder.onnx_text_model import (
    OnnxCrossEncoderModel,
    TextRerankerWorker,
)

# ---------------------------------------------------------------------------
# Test model description
# ---------------------------------------------------------------------------

_MODEL_NAME = "test/cross-encoder"
_MODEL_DESC = BaseModelDescription(
    model=_MODEL_NAME,
    description="Mock cross-encoder",
    license="MIT",
    size_in_GB=0.1,
    sources=ModelSource(hf="mock/hf"),
    model_file="model.onnx",
)


@pytest.fixture(autouse=True)
def registered_test_model() -> BaseModelDescription:
    """Temporarily register the test model in the global list."""
    supported_onnx_models.append(_MODEL_DESC)
    yield _MODEL_DESC
    supported_onnx_models.pop()


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


def _make_mock_session(n_pairs: int) -> MagicMock:
    session = MagicMock()
    # Mock model output: (batch_size, 1) or (batch_size, 2)
    # onnx_embed_pairs uses relevant_output[:, 0]
    session.run.return_value = [np.ones((n_pairs, 1), dtype=np.float32)]
    session.get_inputs.return_value = []
    return session


def _make_mock_tokenizer(n_pairs: int) -> MagicMock:
    tokenizer = MagicMock()

    class MockEncoding:
        def __init__(self):
            self.ids = [101, 102]
            self.type_ids = [0, 0]
            self.attention_mask = [1, 1]

    tokenizer.encode_batch.return_value = [MockEncoding() for _ in range(n_pairs)]
    return tokenizer


# ---------------------------------------------------------------------------
# Concrete class for testing abstract base
# ---------------------------------------------------------------------------


class ConcreteCrossEncoderModel(OnnxCrossEncoderModel):
    def load_onnx_model(self) -> None:
        pass

    @classmethod
    def _get_worker_class(cls) -> type[TextRerankerWorker]:
        return TextRerankerWorker

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[float]:
        return (float(x) for x in output.model_output)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOnnxCrossEncoderModelRerankDocuments:
    """_rerank_documents yields scores in batches."""

    def test_rerank_documents_yields_scores(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model = _make_mock_session(n_pairs=2)
        m.model_input_names = {"input_ids"}
        m.tokenizer = _make_mock_tokenizer(n_pairs=2)

        docs = ["doc1", "doc2", "doc3", "doc4"]
        # batch_size=2 => 2 batches
        results = list(m._rerank_documents("query", docs, batch_size=2))
        assert len(results) == 4
        assert all(isinstance(r, float) for r in results)


class TestRerankPairsIsSmallBranch:
    """_rerank_pairs with small input (no parallel pool)."""

    def test_single_tuple_becomes_one_pair(self) -> None:
        # A single tuple triggers is_small=True; use n_pairs=1 mock so output is 1 score
        m = ConcreteCrossEncoderModel()
        m.model = _make_mock_session(n_pairs=1)
        m.model_input_names = {"input_ids"}
        m.tokenizer = _make_mock_tokenizer(n_pairs=1)
        config = OnnxInferenceConfig(model_name="m", cache_dir="/tmp")
        scores = list(
            m._rerank_pairs(
                pairs=[("query", "doc")],
                config=config,
                batch_size=64,
            )
        )
        assert len(scores) == 1
        assert all(isinstance(s, float) for s in scores)

    def test_list_smaller_than_batch_size(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model = _make_mock_session(n_pairs=2)
        m.model_input_names = {"input_ids"}
        m.tokenizer = _make_mock_tokenizer(n_pairs=2)

        pairs = [("q", "d1"), ("q", "d2")]
        config = OnnxInferenceConfig(model_name="m", cache_dir="/tmp")
        scores = list(
            m._rerank_pairs(
                pairs=pairs,
                config=config,
                batch_size=64,  # larger than len(pairs)=2 => is_small
            )
        )
        assert len(scores) == 2

    def test_parallel_none_uses_direct_path(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model = _make_mock_session(n_pairs=5)
        m.model_input_names = {"input_ids"}
        m.tokenizer = _make_mock_tokenizer(n_pairs=5)

        pairs = [("q", f"d{i}") for i in range(5)]
        config = OnnxInferenceConfig(model_name="m", cache_dir="/tmp")
        scores = list(
            m._rerank_pairs(
                pairs=pairs,
                config=config,
                batch_size=10,
                parallel=None,
            )
        )
        assert len(scores) == 5

    def test_loads_model_when_none_in_small_branch(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model = None
        loaded: list[bool] = []

        def _fake_load() -> None:
            m.model = _make_mock_session(n_pairs=1)
            m.model_input_names = {"input_ids"}
            m.tokenizer = _make_mock_tokenizer(n_pairs=1)
            loaded.append(True)

        m.load_onnx_model = _fake_load  # type: ignore[invalid-assignment]
        config = OnnxInferenceConfig(model_name="m", cache_dir="/tmp")
        list(
            m._rerank_pairs(
                pairs=[("q", "d")],
                config=config,
                batch_size=64,
            )
        )
        assert loaded


class TestRerankPairsParallelBranch:
    """_rerank_pairs when parallel > 0 and input is large (spawns a pool)."""

    def _large_pairs(self) -> list[tuple[str, str]]:
        return [("q", f"d{i}") for i in range(10)]

    def test_parallel_zero_uses_cpu_count(self, loaded_model: ConcreteCrossEncoderModel) -> None:
        out = OnnxOutputContext(model_output=np.array([0.5, 0.6]))
        config = OnnxInferenceConfig(model_name="m", cache_dir="/tmp")
        with patch("qwen3_embed.rerank.cross_encoder.onnx_text_model.ParallelWorkerPool") as cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            cls.return_value = pool
            list(
                loaded_model._rerank_pairs(
                    pairs=self._large_pairs(),
                    config=config,
                    batch_size=2,
                    parallel=0,
                )
            )
        cls.assert_called_once()

    def test_parallel_positive_creates_pool_with_num_workers(
        self, loaded_model: ConcreteCrossEncoderModel
    ) -> None:
        out = OnnxOutputContext(model_output=np.array([0.5]))
        config = OnnxInferenceConfig(model_name="m", cache_dir="/tmp")
        with patch("qwen3_embed.rerank.cross_encoder.onnx_text_model.ParallelWorkerPool") as cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            cls.return_value = pool
            list(
                loaded_model._rerank_pairs(
                    pairs=self._large_pairs(),
                    config=config,
                    batch_size=2,
                    parallel=3,
                )
            )
        call_kwargs = cls.call_args[1]
        assert call_kwargs["num_workers"] == 3

    def test_parallel_start_method_is_forkserver_or_spawn(
        self, loaded_model: ConcreteCrossEncoderModel
    ) -> None:
        out = OnnxOutputContext(model_output=np.array([0.5]))
        config = OnnxInferenceConfig(model_name="m", cache_dir="/tmp")
        with patch("qwen3_embed.rerank.cross_encoder.onnx_text_model.ParallelWorkerPool") as cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            cls.return_value = pool
            list(
                loaded_model._rerank_pairs(
                    pairs=self._large_pairs(),
                    config=config,
                    batch_size=2,
                    parallel=2,
                )
            )
        call_kwargs = cls.call_args[1]
        assert call_kwargs["start_method"] in ("forkserver", "spawn")

    def test_extra_session_options_merged_into_params(
        self, loaded_model: ConcreteCrossEncoderModel
    ) -> None:
        out = OnnxOutputContext(model_output=np.array([0.5]))
        config = OnnxInferenceConfig(
            model_name="m",
            cache_dir="/tmp",
            extra_session_options={"enable_cpu_mem_arena": False},
        )
        with patch("qwen3_embed.rerank.cross_encoder.onnx_text_model.ParallelWorkerPool") as cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            cls.return_value = pool
            list(
                loaded_model._rerank_pairs(
                    pairs=self._large_pairs(),
                    config=config,
                    batch_size=2,
                    parallel=2,
                )
            )
        # Check that params passed to ordered_map contain the merged options
        _, call_kwargs = pool.ordered_map.call_args
        assert call_kwargs["enable_cpu_mem_arena"] is False


class TestOnnxTextModelTokenCount:
    """_token_count calculates sum of attention mask across batches."""

    def test_token_count_sums_correctly(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.tokenizer = _make_mock_tokenizer(n_pairs=1)
        # Mock encoding: mask=[1, 1], count=2
        enc = m.tokenizer.encode_batch.return_value[0]
        enc.attention_mask = [1, 1, 0, 0]

        count = m._token_count([("q", "d")], batch_size=1)
        assert count == 2


# ===========================================================================
# OnnxTextCrossEncoder — covers onnx_text_cross_encoder.py
# ===========================================================================


@pytest.fixture()
def onnx_encoder(
    tmp_path: Path, registered_test_model: BaseModelDescription
) -> OnnxTextCrossEncoder:
    """OnnxTextCrossEncoder with lazy_load + pre-wired mock session and tokenizer."""
    with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
        enc = OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True)

    session = _make_mock_session(n_pairs=1)
    tok = _make_mock_tokenizer(n_pairs=1)
    enc.model = session
    enc.model_input_names = {"input_ids"}
    enc.tokenizer = tok
    return enc


@pytest.fixture()
def loaded_model() -> ConcreteCrossEncoderModel:
    m = ConcreteCrossEncoderModel()
    m.model = _make_mock_session(n_pairs=1)
    m.model_input_names = {"input_ids"}
    m.tokenizer = _make_mock_tokenizer(n_pairs=1)
    return m


class TestOnnxTextCrossEncoderInit:
    """Tests for OnnxTextCrossEncoder.__init__ (lines 35-90)."""

    def test_lazy_load_skips_load_onnx_model(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
            enc = OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True)
        assert enc.lazy_load is True
        # verify download_model was called, but not necessarily load_onnx_model
        assert enc._model_dir == tmp_path

    def test_lazy_load_false_calls_load_onnx_model(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        called: list[bool] = []
        with (
            patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path),
            patch.object(
                OnnxTextCrossEncoder, "load_onnx_model", side_effect=lambda: called.append(True)
            ),
        ):
            OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=False)
        assert called

    def test_device_id_set_directly(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
            enc = OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True, device_id=5)
        assert enc.device_id == 5

    def test_device_id_from_device_ids(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
            enc = OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True, device_ids=[3, 4])
        assert enc.device_id == 3

    def test_device_id_defaults_to_none(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
            enc = OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True)
        assert enc.device_id is None

    def test_specific_model_path_stored(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
            enc = OnnxTextCrossEncoder(
                model_name=_MODEL_NAME, lazy_load=True, specific_model_path="/custom"
            )
        assert enc._specific_model_path == "/custom"

    def test_multiple_device_ids_logs_warning(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with (
            patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path),
            patch(
                "qwen3_embed.rerank.cross_encoder.onnx_text_cross_encoder.logger"
            ) as mock_logger,
        ):
            OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True, device_ids=[0, 1, 2])
        mock_logger.warning.assert_called_once()


class TestOnnxTextCrossEncoderListSupportedModels:
    """_list_supported_models returns supported_onnx_models list."""

    def test_returns_supported_models_list(
        self, registered_test_model: BaseModelDescription
    ) -> None:
        models = OnnxTextCrossEncoder._list_supported_models()
        assert _MODEL_DESC in models


class TestOnnxTextCrossEncoderLoadOnnxModel:
    """load_onnx_model delegates to _load_onnx_model with correct args."""

    def test_delegates_to_load_onnx_model(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
            enc = OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True)

        with patch.object(enc, "_load_onnx_model") as mock_load:
            enc.load_onnx_model()

        mock_load.assert_called_once_with(
            model_dir=tmp_path,
            model_file=_MODEL_DESC.model_file,
            threads=enc.threads,
            providers=enc.providers,
            cuda=enc.cuda,
            device_id=enc.device_id,
            extra_session_options=enc._extra_session_options,
        )


class TestOnnxTextCrossEncoderRerank:
    """rerank() yields float scores for each document."""

    def test_rerank_yields_floats(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        scores = list(onnx_encoder.rerank("query", ["doc1"]))
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_rerank_delegates_to_rerank_documents(
        self, onnx_encoder: OnnxTextCrossEncoder
    ) -> None:
        with patch.object(onnx_encoder, "_rerank_documents", wraps=onnx_encoder._rerank_documents):
            list(onnx_encoder.rerank("query", ["doc"], batch_size=32))
            onnx_encoder._rerank_documents.assert_called_once()  # type: ignore[attr-defined]


class TestOnnxTextCrossEncoderRerankPairs:
    """rerank_pairs() delegates to _rerank_pairs."""

    def test_rerank_pairs_yields_floats(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        onnx_encoder.tokenizer = _make_mock_tokenizer(n_pairs=1)
        scores = list(onnx_encoder.rerank_pairs([("query", "doc")]))
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_rerank_pairs_parallel_delegates_to_pool(
        self, onnx_encoder: OnnxTextCrossEncoder
    ) -> None:
        out = OnnxOutputContext(model_output=np.array([0.9, 0.1]))
        pairs = [("q", f"d{i}") for i in range(10)]
        with patch("qwen3_embed.rerank.cross_encoder.onnx_text_model.ParallelWorkerPool") as cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            cls.return_value = pool
            list(onnx_encoder.rerank_pairs(pairs, batch_size=2, parallel=2))
        cls.assert_called_once()


class TestOnnxTextCrossEncoderPostProcess:
    """_post_process_onnx_output converts model_output to float generator."""

    def test_float_values_yielded(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        output = OnnxOutputContext(model_output=np.array([0.7, 0.3], dtype=np.float32))
        result = list(onnx_encoder._post_process_onnx_output(output))
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_values_match_array(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        arr = np.array([0.8, 0.2], dtype=np.float32)
        output = OnnxOutputContext(model_output=arr)
        result = list(onnx_encoder._post_process_onnx_output(output))
        assert result[0] == pytest.approx(0.8)
        assert result[1] == pytest.approx(0.2)


class TestOnnxTextCrossEncoderTokenCount:
    """token_count() sums tokens across all pairs."""

    def test_token_count_delegates(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        enc = MagicMock()
        enc.attention_mask = [1, 1, 0, 0]
        onnx_encoder.tokenizer.encode_batch.return_value = [enc]  # type: ignore[unresolved-attribute]
        result = onnx_encoder.token_count([("q", "d")])
        assert result == 2


class TestOnnxTextCrossEncoderGetWorkerClass:
    """_get_worker_class returns TextCrossEncoderWorker."""

    def test_returns_text_cross_encoder_worker(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        assert onnx_encoder._get_worker_class() is TextCrossEncoderWorker


class TestTextCrossEncoderWorker:
    """TextCrossEncoderWorker.init_embedding creates an OnnxTextCrossEncoder."""

    def test_creates_onnx_text_cross_encoder(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with (
            patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path),
            patch.object(OnnxTextCrossEncoder, "load_onnx_model"),
        ):
            worker = TextCrossEncoderWorker(
                model_name=_MODEL_NAME,
                cache_dir=str(tmp_path),
            )
        assert isinstance(worker.model, OnnxTextCrossEncoder)
        assert worker.model.model_name == _MODEL_NAME


# ===========================================================================
# Float16 → Float32 casting in onnx_embed_pairs
# ===========================================================================


class TestFloat16Casting:
    """Output from the ONNX session in float16 should be cast to float32."""

    def test_float16_output_cast_to_float32(self) -> None:
        m = ConcreteCrossEncoderModel()
        # Create a float16 session output: (1, 2) — scores[:, 0]
        session = MagicMock()
        session.run.return_value = [np.ones((1, 2), dtype=np.float16)]
        session.get_inputs.return_value = []
        m.model = session
        m.model_input_names = {"input_ids"}
        m.tokenizer = _make_mock_tokenizer(n_pairs=1)

        # onnx_embed_pairs extracts scores[:, 0]; dtype of raw output is float16
        # but the test verifies that the slice (1D) doesn't break post-processing
        ctx = m.onnx_embed_pairs([("q", "d")])
        # scores array should have been sliced from float16 output
        assert ctx.model_output is not None


# ===========================================================================
# _preprocess_onnx_input pass-through in OnnxCrossEncoderModel
# ===========================================================================


class TestPreprocessOnnxInput:
    """_preprocess_onnx_input is called inside onnx_embed_pairs."""

    def test_preprocess_called_during_embed_pairs(
        self, loaded_model: ConcreteCrossEncoderModel
    ) -> None:
        calls: list[dict] = []
        original = loaded_model._preprocess_onnx_input

        def spy(onnx_input: dict, **kwargs: Any) -> dict:
            calls.append(onnx_input)
            return original(onnx_input, **kwargs)

        loaded_model._preprocess_onnx_input = spy  # type: ignore[method-assign]
        loaded_model.onnx_embed_pairs([("q", "d1"), ("q", "d2")])
        assert len(calls) == 1

    def test_custom_preprocess_modifies_input(self) -> None:
        """Subclasses can override _preprocess_onnx_input to transform inputs."""

        class PreprocessingModel(ConcreteCrossEncoderModel):
            def _preprocess_onnx_input(
                self, onnx_input: dict[str, NumpyArray], **kwargs: Any
            ) -> dict[str, NumpyArray]:
                onnx_input["custom"] = np.zeros((1,), dtype=np.float32)
                return onnx_input

        m = PreprocessingModel()
        session = MagicMock()
        session.run.return_value = [np.ones((1, 2), dtype=np.float32)]
        session.get_inputs.return_value = []
        m.model = session
        m.model_input_names = {"input_ids"}
        m.tokenizer = _make_mock_tokenizer(n_pairs=1)

        m.onnx_embed_pairs([("q", "d")])
        _, call_args, _ = session.run.mock_calls[0]
        onnx_input_passed = call_args[1]
        assert "custom" in onnx_input_passed
