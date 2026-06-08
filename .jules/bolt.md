## 2025-05-15 - Redundant List Iteration and NumPy Array Creation
**Learning:** When converting tokenized outputs into NumPy arrays for model input, using list comprehensions with the final `dtype` specified directly (e.g., `np.array([e.ids for e in encoded], dtype=np.int64)`) is the most efficient pattern. It avoids redundant intermediate array creation and is significantly faster than using manual loops with `list.append()`.
**Action:** Refactored `onnx_embed` and `_build_onnx_input` across embedding and reranking modules to use direct `dtype` conversion in list comprehensions.
