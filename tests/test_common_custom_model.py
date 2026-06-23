from unittest.mock import patch

import pytest

from qwen3_embed.common.custom_model import CustomModelSpec, CustomRerankerSpec
from qwen3_embed.common.model_description import (
    BaseModelDescription,
    CustomDenseModelDescription,
    PoolingType,
)


def test_custom_model_spec_register_hf():
    with patch("qwen3_embed.text.text_embedding.TextEmbedding.add_custom_model") as mock_add:
        spec = CustomModelSpec(
            model_id="test-model",
            hf="test-org/test-model",
            dim=768,
            pooling="CLS",
            normalization=True,
            additional_files=["extra.json"],
        )
        spec.register()

        mock_add.assert_called_once()
        args, kwargs = mock_add.call_args
        description = args[0]

        assert isinstance(description, CustomDenseModelDescription)
        assert description.model == "test-model"
        assert description.dim == 768
        assert description.sources.hf == "test-org/test-model"
        assert description.sources.url is None
        assert description.additional_files == ["extra.json"]
        assert kwargs["pooling"] == PoolingType.CLS
        assert kwargs["normalization"] is True


def test_custom_model_spec_register_url():
    with patch("qwen3_embed.text.text_embedding.TextEmbedding.add_custom_model") as mock_add:
        spec = CustomModelSpec(
            model_id="url-model",
            url="https://example.com/model.tar.gz",
            dim=128,
            pooling=PoolingType.MEAN,
            normalization=False,
        )
        spec.register()

        mock_add.assert_called_once()
        args, kwargs = mock_add.call_args
        description = args[0]

        assert description.model == "url-model"
        assert description.sources.url == "https://example.com/model.tar.gz"
        assert description.sources.hf is None
        assert kwargs["pooling"] == PoolingType.MEAN
        assert kwargs["normalization"] is False


def test_custom_model_spec_requires_dim():
    spec = CustomModelSpec(model_id="no-dim", hf="org/model")
    with pytest.raises(ValueError, match="dim is required"):
        spec.register()


def test_custom_reranker_spec_register_hf():
    with patch(
        "qwen3_embed.rerank.cross_encoder.text_cross_encoder.TextCrossEncoder.add_custom_model"
    ) as mock_add:
        spec = CustomRerankerSpec(
            model_id="test-reranker",
            hf="test-org/test-reranker",
            model_file="custom.onnx",
            description="A test reranker",
            license="Apache-2.0",
            size_in_GB=0.5,
            additional_files=["config.json"],
        )
        spec.register()

        mock_add.assert_called_once()
        args, _ = mock_add.call_args
        description = args[0]

        assert isinstance(description, BaseModelDescription)
        assert description.model == "test-reranker"
        assert description.sources.hf == "test-org/test-reranker"
        assert description.model_file == "custom.onnx"
        assert description.description == "A test reranker"
        assert description.license == "Apache-2.0"
        assert description.size_in_GB == 0.5
        assert description.additional_files == ["config.json"]


def test_custom_reranker_spec_register_url():
    with patch(
        "qwen3_embed.rerank.cross_encoder.text_cross_encoder.TextCrossEncoder.add_custom_model"
    ) as mock_add:
        spec = CustomRerankerSpec(
            model_id="url-reranker", url="https://example.com/reranker.tar.gz"
        )
        spec.register()

        mock_add.assert_called_once()
        args, _ = mock_add.call_args
        description = args[0]

        assert description.model == "url-reranker"
        assert description.sources.url == "https://example.com/reranker.tar.gz"
        assert description.sources.hf is None
