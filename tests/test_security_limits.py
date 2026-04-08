import pytest

# Needs to be set BEFORE modules are imported to affect MAX_INPUT_LENGTH
import qwen3_embed.common.utils
import qwen3_embed.rerank.cross_encoder.text_cross_encoder
import qwen3_embed.text.text_embedding

qwen3_embed.common.utils.MAX_INPUT_LENGTH = 100


def test_check_input_length():
    qwen3_embed.common.utils.check_input_length("a" * 100)
    with pytest.raises(ValueError):
        qwen3_embed.common.utils.check_input_length("a" * 101)


def test_iter_checked_texts():
    texts = ["a" * 10, "b" * 100, "c" * 5]
    assert list(qwen3_embed.common.utils.iter_checked_texts(texts)) == texts
    iterator = qwen3_embed.common.utils.iter_checked_texts(["a" * 10, "b" * 101])
    assert next(iterator) == "a" * 10
    with pytest.raises(ValueError):
        next(iterator)


class MockModel:
    def embed(self, docs, *args, **kwargs):
        docs = list(docs)
        yield from [1 for _ in docs]

    def query_embed(self, q, *args, **kwargs):
        q = list(q)
        yield from [1 for _ in q]

    def passage_embed(self, texts, *args, **kwargs):
        texts = list(texts)
        yield from [1 for _ in texts]

    def rerank(self, query, docs, *args, **kwargs):
        docs = list(docs)
        yield from [1 for _ in docs]

    def rerank_pairs(self, pairs, *args, **kwargs):
        pairs = list(pairs)
        yield from [1 for _ in pairs]


def test_text_embedding_limits(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(
            qwen3_embed.text.text_embedding.TextEmbedding,
            "__init__",
            lambda self, *args, **kwargs: setattr(self, "model", MockModel()),
        )
        te = qwen3_embed.text.text_embedding.TextEmbedding("dummy")
        with pytest.raises(ValueError):
            list(te.embed("a" * 101))
        with pytest.raises(ValueError):
            list(te.embed(["a" * 50, "b" * 101]))
        assert len(list(te.embed(["a" * 10, "b" * 20]))) == 2
        with pytest.raises(ValueError):
            list(te.query_embed("a" * 101))
        with pytest.raises(ValueError):
            list(te.passage_embed(["a" * 10, "b" * 101]))


def test_text_cross_encoder_limits(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(
            qwen3_embed.rerank.cross_encoder.text_cross_encoder.TextCrossEncoder,
            "__init__",
            lambda self, *args, **kwargs: setattr(self, "model", MockModel()),
        )
        tce = qwen3_embed.rerank.cross_encoder.text_cross_encoder.TextCrossEncoder("dummy")
        with pytest.raises(ValueError):
            list(tce.rerank("a" * 101, ["doc1"]))
        with pytest.raises(ValueError):
            list(tce.rerank("query", ["doc1", "b" * 101]))
        with pytest.raises(ValueError):
            list(tce.rerank_pairs([("q1", "d1"), ("q2", "d2" * 101)]))
        assert len(list(tce.rerank("query", ["doc1", "doc2"]))) == 2
        assert len(list(tce.rerank_pairs([("q1", "d1"), ("q2", "d2")]))) == 2
