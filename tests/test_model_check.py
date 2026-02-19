import unittest.mock as mock

import pytest

from qwen3_embed.text.onnx_embedding import OnnxTextEmbedding


def test_onnx_embed_raises_when_model_not_loaded():
    # lazy_load=True means load_onnx_model() is NOT called in __init__
    # Use a real model name that exists or is default, but since we don't load it, it might not matter
    # if it doesn't try to download it.
    # OnnxTextEmbedding constructor calls download_model.
    # To avoid downloading, we might need to mock or use local files.
    # But lazy_load=True prevents load_onnx_model(), but download_model happens in __init__.

    # Let's try to mock the download or use a model that is already cached or minimal.
    # If the environment has cached models, it will be fast.
    # If not, it might try to download.

    # We can use a non-existent model name and catch the error if download fails?
    # No, download_model happens before lazy_load check.

    # If we want to avoid side effects (downloading), we should mock download_model.
    pass


def test_onnx_embed_raises_without_loading():
    with mock.patch(
        "qwen3_embed.text.onnx_embedding.OnnxTextEmbedding.download_model"
    ) as mock_download:
        mock_download.return_value = "mock_path"

        # We also need to mock _get_model_description because it's called in __init__
        with (
            mock.patch(
                "qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._get_model_description"
            ) as mock_desc,
        ):
            mock_desc.return_value = mock.Mock()
            mock_desc.return_value.model_file = "model.onnx"

            model = OnnxTextEmbedding(model_name="mock-model", lazy_load=True)

            with pytest.raises(ValueError, match="Model not loaded"):
                model.onnx_embed(["test"])
