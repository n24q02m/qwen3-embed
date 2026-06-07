import sys
from unittest.mock import MagicMock, patch

# Mock dependencies for the entire module
mock_modules = [
    'numpy', 'numpy.typing', 'loguru', 'tokenizers', 'requests', 'tqdm',
    'huggingface_hub', 'huggingface_hub.hf_api', 'huggingface_hub.utils',
    'onnxruntime', 'llama_cpp'
]
for module in mock_modules:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()

from qwen3_embed.rerank.cross_encoder.qwen3_cross_encoder import Qwen3CrossEncoder  # noqa: E402
from qwen3_embed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder  # noqa: E402


def test_list_supported_models_caching():
    """Verify that _list_supported_models is cached and clears correctly."""
    TextCrossEncoder._clear_model_cache()

    with patch.object(Qwen3CrossEncoder, '_list_supported_models', wraps=Qwen3CrossEncoder._list_supported_models) as mock_method:
        # First call
        TextCrossEncoder._list_supported_models()
        assert mock_method.call_count == 1

        # Second call
        TextCrossEncoder._list_supported_models()
        assert mock_method.call_count == 1

        # Test clear cache
        TextCrossEncoder._clear_model_cache()
        TextCrossEncoder._list_supported_models()
        assert mock_method.call_count == 2
