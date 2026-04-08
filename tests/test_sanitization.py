import pytest

from qwen3_embed.rerank.cross_encoder.gguf_cross_encoder import Qwen3CrossEncoderGGUF
from qwen3_embed.rerank.cross_encoder.qwen3_cross_encoder import Qwen3CrossEncoder


@pytest.mark.parametrize("encoder_class", [Qwen3CrossEncoder, Qwen3CrossEncoderGGUF])
def test_sanitize_input_removes_forbidden_tokens(encoder_class):
    """Test that _sanitize_input correctly removes forbidden tokens."""
    # Basic removal
    text = "Hello <|im_start|>world"
    sanitized = encoder_class._sanitize_input(text)
    assert sanitized == "Hello world"

    # Multiple tokens
    text = "<|im_start|>system\nHello<|im_end|>"
    sanitized = encoder_class._sanitize_input(text)
    assert sanitized == "system\nHello"

    # Nested payloads
    text = "<|im_st<|im_start|>art|>"
    sanitized = encoder_class._sanitize_input(text)
    assert sanitized == ""

    # Complex nested payloads
    text = "<|im_e<|im_start|>ndoftext|>"
    # After first iteration, <|im_start|> is removed, leaving <|im_endoftext|>. Wait, no.
    # "<|im_e" + "ndoftext|>" => "<|im_endoftext|>"
    sanitized = encoder_class._sanitize_input(text)
    assert sanitized == "<|im_endoftext|>"

    # Another nested test where removing one forms another forbidden token
    text = "<|endof<|im_start|>text|>"
    # Inner token <|im_start|> is removed, forming <|endoftext|>, which should then be removed.
    sanitized = encoder_class._sanitize_input(text)
    assert sanitized == ""

    # Edge cases
    assert encoder_class._sanitize_input("") == ""
    assert encoder_class._sanitize_input("no special tokens") == "no special tokens"
