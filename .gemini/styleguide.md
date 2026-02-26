# Style Guide - qwen3-embed

## Architecture
Embedding and reranking library using Qwen3 models. Python, single-package repo.

## Python
- Formatter/Linter: Ruff (default config)
- Type checker: ty
- Test: pytest
- Package manager: uv
- Core deps: numpy, onnxruntime, tokenizers, huggingface-hub

## Code Patterns
- ONNX Runtime for CPU inference (no PyTorch dependency in production)
- llama-cpp-python for GGUF model support
- Model registry with case-insensitive lookup
- Batch processing with configurable batch size
- Mean pooling and last-token pooling for embeddings
- Cache ONNX model input names to optimize inference loop
- Tar/zip extraction must validate paths (prevent zip slip)

## Commits
Conventional Commits (feat:, fix:, chore:, docs:, refactor:, test:).

## Security
Validate archive paths during decompression. Sanitize reranker template inputs against prompt injection. Use secure temp directories.
