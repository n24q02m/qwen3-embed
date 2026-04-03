import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock llama_cpp for initial import
mock_llama_module = MagicMock()
sys.modules["llama_cpp"] = mock_llama_module

from qwen3_embed.text.gguf_embedding import Qwen3TextEmbeddingGGUF  # noqa: E402


def _make_model(embedding_dim=4):
    """Create a GGUF model instance with mocked dependencies."""
    mock_llama_cls = MagicMock()
    mock_llm = MagicMock()
    mock_llama_cls.return_value = mock_llm

    # Mock create_embedding to return vectors for each input
    def mock_create_embedding(docs, *args, **kwargs):
        return {"data": [{"embedding": [1.0] * embedding_dim} for _ in docs]}

    mock_llm.create_embedding.side_effect = mock_create_embedding

    with (
        patch.dict(sys.modules, {"llama_cpp": mock_llama_module}),
        patch.object(Qwen3TextEmbeddingGGUF, "download_model", return_value="/tmp"),
        patch("qwen3_embed.text.gguf_embedding.define_cache_dir", return_value=Path("/tmp")),
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_llama_module.Llama = mock_llama_cls
        model = Qwen3TextEmbeddingGGUF(model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF")
        return model


def test_embed_batch_calls():
    """Verify that calling embed with 5 docs and batch_size 2 results in 3 create_embedding calls."""
    model = _make_model(embedding_dim=4)
    docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]

    # Consume the generator
    results = list(model.embed(docs, batch_size=2))

    assert len(results) == 5
    assert model._llm.create_embedding.call_count == 3

    # Verify call arguments
    calls = [call[0][0] for call in model._llm.create_embedding.call_args_list]
    assert calls[0] == ["doc1", "doc2"]
    assert calls[1] == ["doc3", "doc4"]
    assert calls[2] == ["doc5"]


def test_embed_single_batch():
    """Verify that calling embed with 3 docs and batch_size 5 results in 1 create_embedding call."""
    model = _make_model(embedding_dim=4)
    docs = ["doc1", "doc2", "doc3"]

    results = list(model.embed(docs, batch_size=5))

    assert len(results) == 3
    assert model._llm.create_embedding.call_count == 1
    assert model._llm.create_embedding.call_args[0][0] == ["doc1", "doc2", "doc3"]
