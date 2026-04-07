import re

import pytest

from qwen3_embed.rerank.cross_encoder.gguf_cross_encoder import Qwen3CrossEncoderGGUF
from qwen3_embed.rerank.cross_encoder.qwen3_cross_encoder import Qwen3CrossEncoder


@pytest.mark.parametrize("encoder_cls", [Qwen3CrossEncoderGGUF, Qwen3CrossEncoder])
@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Hello world", "Hello world"),
        ("<|im_start|>Hello world", "Hello world"),
        ("<|im_start|>Hello<|im_end|>", "Hello"),
        ("<|im_start|><|im_start|>Nested<|im_end|><|im_end|>", "Nested"),
        ("<|<|im_start|>im_start|>Deconstructed", "Deconstructed"),
        ("<|endoftext|>End of text", "End of text"),
        (
            "Multiple <|im_start|>tokens<|im_end|> and <|endoftext|> more.",
            "Multiple tokens and  more.",
        ),
        ("", ""),
        ("   ", "   "),
    ],
)
def test_sanitize_input(encoder_cls, input_text, expected_output):
    assert encoder_cls._sanitize_input(input_text) == expected_output


def test_regex_logic():
    # Test the proposed regex logic in isolation
    FORBIDDEN_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    _FORBIDDEN_PATTERN = re.compile("|".join(re.escape(token) for token in FORBIDDEN_TOKENS))

    def sanitize(text):
        while True:
            text, n = _FORBIDDEN_PATTERN.subn("", text)
            if n == 0:
                break
        return text

    assert sanitize("<|<|im_start|>im_start|>") == ""
    assert sanitize("<|im_start|>system<|im_end|>") == "system"
