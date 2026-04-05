import pytest

from qwen3_embed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase


class StubTextCrossEncoder(TextCrossEncoderBase):
    """A concrete subclass of TextCrossEncoderBase for testing."""

    pass


def test_initialization():
    """Test that TextCrossEncoderBase initializes correctly."""
    model_name = "test-model"
    cache_dir = "/tmp/cache"
    threads = 4
    encoder = StubTextCrossEncoder(
        model_name=model_name, cache_dir=cache_dir, threads=threads, local_files_only=True
    )

    assert encoder.model_name == model_name
    assert encoder.cache_dir == cache_dir
    assert encoder.threads == threads
    assert encoder._local_files_only is True


def test_initialization_defaults():
    """Test initialization with default values."""
    encoder = StubTextCrossEncoder(model_name="test-model")

    assert encoder.model_name == "test-model"
    assert encoder.cache_dir is None
    assert encoder.threads is None
    assert encoder._local_files_only is False


def test_rerank_raises_not_implemented():
    """Test that rerank raises NotImplementedError."""
    encoder = StubTextCrossEncoder(model_name="test-model")
    with pytest.raises(
        NotImplementedError, match="This method should be overridden by subclasses"
    ):
        list(encoder.rerank(query="test", documents=["doc1"]))


def test_rerank_pairs_raises_not_implemented():
    """Test that rerank_pairs raises NotImplementedError."""
    encoder = StubTextCrossEncoder(model_name="test-model")
    with pytest.raises(
        NotImplementedError, match="This method should be overridden by subclasses"
    ):
        list(encoder.rerank_pairs(pairs=[("q", "d")]))


def test_token_count_raises_not_implemented():
    """Test that token_count raises NotImplementedError."""
    encoder = StubTextCrossEncoder(model_name="test-model")
    with pytest.raises(
        NotImplementedError, match="This method should be overridden by subclasses"
    ):
        encoder.token_count(pairs=[("q", "d")])
