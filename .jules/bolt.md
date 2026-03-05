
## 2024-03-02 - [Optimize Tokenizer Encoding Processing for ONNX Inputs]
**Learning:** When preparing ONNX model inputs from `tokenizers.Encoding` objects, doing multiple list comprehensions (`[e.ids for e in encoded]`) followed by NumPy array conversions (and copying via `np.array(input_ids, dtype=np.int64)`) incurs significant memory allocation overhead.
**Action:** Implemented a single-pass loop over the encoded outputs to populate plain Python lists (`input_ids_list` and `attention_mask_list`), which are then converted directly to `np.int64` NumPy arrays exactly once. This reduces intermediate array copies and speeds up preprocessing logic by ~28% locally.
