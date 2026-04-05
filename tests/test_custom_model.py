"""Tests for custom model registration via TextEmbedding.add_custom_model."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource, PoolingType
from qwen3_embed.text.custom_text_embedding import CustomTextEmbedding
from qwen3_embed.text.text_embedding import TextEmbedding


class TestCustomModelRegistration:
    """Verify custom model registration works for all pooling types."""

    def setup_method(self):
        """Clear custom model registry between tests."""
        CustomTextEmbedding.SUPPORTED_MODELS.clear()
        CustomTextEmbedding.POSTPROCESSING_MAPPING.clear()

    def test_register_cls_pooling_model(self):
        TextEmbedding.add_custom_model(
            model="test/cls-model",
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf="test/cls-model"),
            dim=768,
        )
        models = TextEmbedding.list_supported_models()
        assert any(m["model"] == "test/cls-model" for m in models)

    def test_register_mean_pooling_model(self):
        TextEmbedding.add_custom_model(
            model="test/mean-model",
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf="test/mean-model"),
            dim=512,
        )
        models = TextEmbedding.list_supported_models()
        assert any(m["model"] == "test/mean-model" for m in models)

    def test_register_last_token_pooling_model(self):
        TextEmbedding.add_custom_model(
            model="test/last-token-model",
            pooling=PoolingType.LAST_TOKEN,
            normalization=True,
            sources=ModelSource(hf="test/last-token-model"),
            dim=1024,
        )
        models = TextEmbedding.list_supported_models()
        assert any(m["model"] == "test/last-token-model" for m in models)

    def test_duplicate_model_raises_case_insensitive(self):
        """Verify duplicate model registration is caught regardless of casing."""
        TextEmbedding.add_custom_model(
            model="test/Duplicate-Model",
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf="test/duplicate-model"),
            dim=256,
        )
        with pytest.raises(ValueError, match="already registered"):
            TextEmbedding.add_custom_model(
                model="TEST/duplicate-model",
                pooling=PoolingType.CLS,
                normalization=True,
                sources=ModelSource(hf="test/duplicate-model"),
                dim=256,
            )

    def test_registration_conflicts_with_builtin(self):
        """Verify custom models cannot shadow built-in models."""
        # 'n24q02m/Qwen3-Embedding-0.6B-ONNX' is a known built-in model
        builtin_model = "n24q02m/Qwen3-Embedding-0.6B-ONNX"
        with pytest.raises(ValueError, match="already registered"):
            TextEmbedding.add_custom_model(
                model=builtin_model,
                pooling=PoolingType.CLS,
                normalization=True,
                sources=ModelSource(hf="some/other-model"),
                dim=1024,
            )

    def test_add_custom_model_with_all_params(self):
        """Exercise all parameters of add_custom_model."""
        TextEmbedding.add_custom_model(
            model="test/all-params",
            pooling=PoolingType.CLS,
            normalization=False,
            sources=ModelSource(hf="test/all-params"),
            dim=128,
            model_file="custom/model.onnx",
            description="A test model with all params",
            license="MIT",
            size_in_gb=1.5,
            additional_files=["vocab.txt", "config.json"],
        )
        models = TextEmbedding.list_supported_models()
        matching = [m for m in models if m["model"] == "test/all-params"]
        assert len(matching) == 1
        model = matching[0]
        assert model["model_file"] == "custom/model.onnx"
        assert model["description"] == "A test model with all params"
        assert model["license"] == "MIT"
        assert model["size_in_GB"] == 1.5
        assert model["additional_files"] == ["vocab.txt", "config.json"]

    def test_get_embedding_size_custom_model(self):
        """Verify get_embedding_size works for custom models."""
        TextEmbedding.add_custom_model(
            model="test/size-model",
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf="test/size-model"),
            dim=999,
        )
        assert TextEmbedding.get_embedding_size("test/size-model") == 999
        # Case insensitive
        assert TextEmbedding.get_embedding_size("TEST/SIZE-MODEL") == 999

    def test_get_embedding_size_raises_for_unknown(self):
        """Verify get_embedding_size raises ValueError for unknown models."""
        with pytest.raises(ValueError, match="Available model names"):
            TextEmbedding.get_embedding_size("nonexistent-model")

    def test_initialization_and_delegation(self):
        """Verify initialization and delegation to the underlying custom model."""
        model_name = "test/delegation-model"
        TextEmbedding.add_custom_model(
            model=model_name,
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf=model_name),
            dim=384,
        )

        mock_instance = MagicMock()
        mock_instance.embed.return_value = iter([np.array([0.1])])
        mock_instance.query_embed.return_value = iter([np.array([0.2])])
        mock_instance.passage_embed.return_value = iter([np.array([0.3])])
        mock_instance.token_count.return_value = 42

        MockCustomEmb = MagicMock(return_value=mock_instance)
        MockCustomEmb._list_supported_models.return_value = [
            DenseModelDescription(
                model=model_name,
                sources=ModelSource(hf=model_name),
                dim=384,
                model_file="onnx/model.onnx",
                description="",
                license="",
                size_in_GB=0.0,
            )
        ]

        with patch.object(TextEmbedding, "EMBEDDINGS_REGISTRY", [MockCustomEmb]):
            emb = TextEmbedding(model_name=model_name)

            # Verify initialization called MockCustomEmb with correct args
            MockCustomEmb.assert_called_once()
            _, kwargs = MockCustomEmb.call_args
            assert kwargs['model_name'] == model_name

            # Verify embedding size property
            assert emb.embedding_size == 384

            # Verify delegation of embed
            list(emb.embed("test doc"))
            mock_instance.embed.assert_called_once_with("test doc", 256, None)

            # Verify delegation of query_embed
            list(emb.query_embed("test query"))
            mock_instance.query_embed.assert_called_once_with("test query")

            # Verify delegation of passage_embed
            list(emb.passage_embed(["passage"]))
            mock_instance.passage_embed.assert_called_once_with(["passage"])

            # Verify delegation of token_count
            assert emb.token_count("some text") == 42
            mock_instance.token_count.assert_called_once_with("some text", batch_size=1024)
