import pytest

from qwen3_embed.rerank.cross_encoder.gguf_cross_encoder import Qwen3CrossEncoderGGUF
from qwen3_embed.rerank.cross_encoder.qwen3_cross_encoder import Qwen3CrossEncoder


@pytest.mark.parametrize("encoder_cls", [Qwen3CrossEncoder, Qwen3CrossEncoderGGUF])
class TestSanitization:
    def test_basic_stripping(self, encoder_cls):
        text = "<|im_start|>hello<|im_end|>"
        assert encoder_cls._sanitize_input(text) == "hello"

    def test_nested_stripping(self, encoder_cls):
        # Nested bypass attempt: <|<|im_start|>im_start|> -> should be fully stripped
        text = "<|<|im_start|>im_start|>hello"
        assert encoder_cls._sanitize_input(text) == "hello"

        # Deeper nesting
        text = "<|<|<|im_start|>im_start|>im_start|>hidden"
        assert encoder_cls._sanitize_input(text) == "hidden"

    def test_multiple_tokens(self, encoder_cls):
        text = "<|im_start|>user<|im_end|><|endoftext|>"
        assert encoder_cls._sanitize_input(text) == "user"

    def test_no_forbidden_tokens(self, encoder_cls):
        text = "This is a clean string."
        assert encoder_cls._sanitize_input(text) == text

    def test_partial_match(self, encoder_cls):
        # Should not strip if it is not an exact match
        text = "<|im_st art|>"
        assert encoder_cls._sanitize_input(text) == text
