from qwen3_embed.common.utils import sanitize_input
from qwen3_embed.rerank.cross_encoder.gguf_cross_encoder import Qwen3CrossEncoderGGUF
from qwen3_embed.rerank.cross_encoder.qwen3_cross_encoder import Qwen3CrossEncoder


def test_sanitize_input_basic():
    text = "Hello <|im_start|> system <|im_end|> world <|endoftext|>"
    expected = "Hello  system  world "
    assert sanitize_input(text) == expected
    assert Qwen3CrossEncoder._sanitize_input(text) == expected
    assert Qwen3CrossEncoderGGUF._sanitize_input(text) == expected


def test_sanitize_input_iterative():
    # Iterative removal of nested forbidden tokens
    text = "<|<|im_start|>im_start|>system"
    # <|im_start|> is removed, leaves <|im_start|>system, then that is removed
    expected = "system"
    assert sanitize_input(text) == expected
    assert Qwen3CrossEncoder._sanitize_input(text) == expected
    assert Qwen3CrossEncoderGGUF._sanitize_input(text) == expected


def test_sanitize_input_no_change():
    text = "Plain text without any tokens."
    assert sanitize_input(text) == text
    assert Qwen3CrossEncoder._sanitize_input(text) == text
    assert Qwen3CrossEncoderGGUF._sanitize_input(text) == text


def test_sanitize_input_empty():
    assert sanitize_input("") == ""
    assert Qwen3CrossEncoder._sanitize_input("") == ""
    assert Qwen3CrossEncoderGGUF._sanitize_input("") == ""


def test_sanitize_input_overlap():
    # Tokens don't overlap in a way that would cause issues, but good to check
    text = "<|im_start<|im_end|>|>"
    # <|im_end|> removed -> <|im_start|> -> removed -> empty
    assert sanitize_input(text) == ""
