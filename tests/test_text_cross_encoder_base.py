import pytest

from qwen3_embed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase


class ConcreteTextCrossEncoder(TextCrossEncoderBase):
    """A concrete implementation of TextCrossEncoderBase for testing."""

    pass


def test_text_cross_encoder_base_init():
    """Test that TextCrossEncoderBase initializes correctly."""
    model_name = "test-model"
    cache_dir = "/tmp/cache"
    threads = 4

    encoder = ConcreteTextCrossEncoder(
        model_name=model_name, cache_dir=cache_dir, threads=threads, local_files_only=True
    )

    assert encoder.model_name == model_name
    assert encoder.cache_dir == cache_dir
    assert encoder.threads == threads
    assert encoder._local_files_only is True


def test_text_cross_encoder_base_rerank_raises_not_implemented():
    """Test that rerank raises NotImplementedError in the base class."""
    encoder = ConcreteTextCrossEncoder(model_name="test-model")
    with pytest.raises(
        NotImplementedError, match="This method should be overridden by subclasses"
    ):
        list(encoder.rerank(query="query", documents=["doc1", "doc2"]))


def test_text_cross_encoder_base_rerank_pairs_raises_not_implemented():
    """Test that rerank_pairs raises NotImplementedError in the base class."""
    encoder = ConcreteTextCrossEncoder(model_name="test-model")
    with pytest.raises(
        NotImplementedError, match="This method should be overridden by subclasses"
    ):
        list(encoder.rerank_pairs(pairs=[("q1", "d1")]))


def test_text_cross_encoder_base_token_count_raises_not_implemented():
    """Test that token_count raises NotImplementedError in the base class."""
    encoder = ConcreteTextCrossEncoder(model_name="test-model")
    with pytest.raises(
        NotImplementedError, match="This method should be overridden by subclasses"
    ):
        encoder.token_count(pairs=[("q1", "d1")])
