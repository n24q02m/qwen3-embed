import pytest

from qwen3_embed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase


class StubTextCrossEncoder(TextCrossEncoderBase):
    """A stub implementation of TextCrossEncoderBase for testing."""

    pass


def test_text_cross_encoder_base_init():
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


def test_text_cross_encoder_base_init_defaults():
    encoder = StubTextCrossEncoder(model_name="test-model")
    assert encoder.model_name == "test-model"
    assert encoder.cache_dir is None
    assert encoder.threads is None
    assert encoder._local_files_only is False


def test_text_cross_encoder_base_rerank_raises_not_implemented():
    encoder = StubTextCrossEncoder(model_name="test-model")
    with pytest.raises(
        NotImplementedError, match="This method should be overridden by subclasses"
    ):
        list(encoder.rerank("query", ["doc1"]))


def test_text_cross_encoder_base_rerank_pairs_raises_not_implemented():
    encoder = StubTextCrossEncoder(model_name="test-model")
    with pytest.raises(
        NotImplementedError, match="This method should be overridden by subclasses"
    ):
        list(encoder.rerank_pairs([("query", "doc1")]))


def test_text_cross_encoder_base_token_count_raises_not_implemented():
    encoder = StubTextCrossEncoder(model_name="test-model")
    with pytest.raises(
        NotImplementedError, match="This method should be overridden by subclasses"
    ):
        encoder.token_count([("query", "doc1")])
