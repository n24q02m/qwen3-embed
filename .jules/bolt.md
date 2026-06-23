## 2025-06-23 - Qwen3CrossEncoder Performance Optimization

**Learning:** Qwen3 reranker implementation used list comprehensions to format inputs for multiple documents/pairs before passing them to a method that iterated over them anyway.

**Action:** Replaced list comprehensions with generator expressions in `onnx_embed` and `onnx_embed_pairs` and updated `_onnx_embed_texts` to accept an `Iterable[str]`. This reduces peak memory usage when processing large batches of documents by lazily formatting inputs as they are consumed by the inference loop.
