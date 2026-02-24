"""Unit tests for Qwen3CrossEncoderGGUF chat template formatting."""

from qwen3_embed.rerank.cross_encoder.gguf_cross_encoder import (
    DEFAULT_INSTRUCTION,
    SYSTEM_PROMPT,
    Qwen3CrossEncoderGGUF,
)


class TestQwen3GGUFChatTemplate:
    """Verify chat template formatting for GGUF reranking."""

    def test_format_rerank_input(self):
        """Test standard input formatting."""
        result = Qwen3CrossEncoderGGUF._format_rerank_input(
            query="What is AI?",
            document="AI is artificial intelligence.",
            instruction=DEFAULT_INSTRUCTION,
        )
        assert "<|im_start|>system" in result
        assert SYSTEM_PROMPT in result
        assert "<Instruct>:" in result
        assert "<Query>: What is AI?" in result
        assert "<Document>: AI is artificial intelligence." in result
        assert "<|im_end|>" in result
        assert "<think>" in result
        assert "</think>" in result

    def test_custom_instruction(self):
        """Test with custom instruction."""
        result = Qwen3CrossEncoderGGUF._format_rerank_input(
            query="q",
            document="d",
            instruction="Custom task instruction",
        )
        assert "<Instruct>: Custom task instruction" in result

    def test_empty_inputs(self):
        """Test with empty query and document."""
        result = Qwen3CrossEncoderGGUF._format_rerank_input(
            query="",
            document="",
            instruction=DEFAULT_INSTRUCTION,
        )
        assert "<Query>: \n" in result
        assert "<Document>: <|im_end|>" in result

    def test_special_characters(self):
        """Test with special characters in inputs."""
        query = "Query with <tags> & symbols"
        document = "Document with \n newlines and 'quotes'"
        result = Qwen3CrossEncoderGGUF._format_rerank_input(
            query=query,
            document=document,
            instruction=DEFAULT_INSTRUCTION,
        )
        assert f"<Query>: {query}" in result
        assert f"<Document>: {document}" in result
