## 2026-05-01 - [Fast 2-class classification in Numpy]
**Learning:** For 2-class classification (binary softmax) in numpy, computing the sigmoid on the difference of logits is fast. This can be further optimized by using `np.subtract(yes_no_logits[:, 0], yes_no_logits[:, 1], dtype=np.float32)` instead of casting individually to floats and then subtracting, saving array allocations and execution time. Also, using in-place operations like `np.exp(diff, out=diff)` and `np.reciprocal(diff, out=diff)` avoids extra memory overhead.
**Action:** Always prefer `np.subtract` over `array.astype() - array.astype()` and utilize in-place mutation where possible when optimizing hot paths involving numpy arrays.

## 2024-05-24 - [Fast iterable chunking with walrus operator]
**Learning:** When chunking generic iterables in hot paths using `itertools.islice`, using the walrus operator (`while b := list(islice(source_iter, size)):`) instead of a `while True` loop with an explicit length check reduces bytecode execution overhead.
**Action:** Use the walrus operator for iterator exhaustion loops to avoid unnecessary length checks and loop breaks.

## 2024-05-06 - [Fast single scalar math operations]
**Learning:** For mathematical operations on single scalar values (e.g., computing a sigmoid from a logit difference), using Python's built-in `math.exp` is significantly faster than `numpy.exp` due to the avoidance of numpy's C-API dispatch and object creation overhead. Because `math.exp` raises an `OverflowError` for large negative exponents, it should be wrapped in a `try...except OverflowError` block to handle edge cases appropriately (e.g., returning 0.0 or 1.0 for sigmoid boundaries).
**Action:** Use `math.exp` instead of `np.exp` when operating on single scalar values, wrapping it in a `try...except OverflowError` block to handle numerical boundaries.

## 2024-05-24 - [Fast last token index in right-padded masks]
**Learning:** When calculating the last token index for strictly right-padded attention masks (contiguous 1s followed by 0s) in numpy, using `mask.sum(axis=1) - 1` is significantly faster (~4-5x) than `seq_len - 1 - np.argmax(mask[:, ::-1], axis=1)`. It avoids array reversal and argmax allocation overhead.
**Action:** Use `mask.sum(axis=1) - 1` to find the last valid token index for right-padded attention masks.

## 2025-05-24 - Fast L2 Normalization
**Learning:** In hot paths like embedding normalization, chained numpy operations (`np.sqrt`, `np.maximum`, `/`) allocate large intermediate arrays.
**Action:** Use in-place operations (`out=` parameter for ufuncs and `/=`) to reuse memory. We optimized `normalize` in `utils.py` to mutate the array in-place, yielding a ~25% speedup without breaking semantics, as the pooled array is temporary.

## 2026-05-26 - [Fast logit subtraction without stack]
**Learning:** When subtracting two columns from a 2D numpy array (e.g. logit extraction), constructing an intermediate array via `np.stack` creates unnecessary allocation overhead. Direct subtraction of the sliced columns via `np.subtract(last_logits[:, NO], last_logits[:, YES])` avoids this.
**Action:** Avoid `np.stack` for simple column extractions when the immediate next step is a reduction or subtraction operation.

## 2026-06-10 - [Fast NumPy Input Tensor Creation]
**Learning:** Creating NumPy arrays from a sequence of lists (e.g., tokenized input) with a specified `dtype` (e.g., `dtype=np.int64`) in a single call is more efficient than creating a default array first and then casting it. The latter leads to redundant O(N) memory allocations and copies because `np.array(..., dtype=np.int64)` on an existing array defaults to `copy=True`.
**Action:** Always specify the final `dtype` when creating input tensors for model inference from Python sequences to avoid intermediate copies.

## 2026-06-11 - Optimize last_token_pool using reverse argmax
**Learning:** Finding the last non-zero token index in a padding mask using `seq_len - 1 - np.argmax(attention_mask[:, ::-1], axis=1)` is significantly (~2.5x) faster than the previous `np.argmax(np.cumsum(attention_mask, axis=1), axis=1)` approach. `cumsum` does unnecessary arithmetic over the entire sequence dimension, whereas `argmax` over the reversed array operates much more efficiently.
**Action:** Use reverse argmax to find the last occurrence of a condition in numpy arrays rather than cumsum-based strategies when dealing with boolean/binary masks.

## 2024-05-27 - [Fast stream reordering with sentinel pop]
**Learning:** When reordering items from a concurrent generator stream (where items arrive out-of-order and must be yielded sequentially), implementing a "fast path" that immediately yields items arriving in the correct expected order bypasses the overhead of dictionary insertion. Furthermore, using `dict.pop(key, sentinel)` combined with a `while` loop allows simultaneous existence-checking and extraction, avoiding double lookups (`key in dict` followed by `dict.pop(key)`).
**Action:** When buffering out-of-order stream results in a dictionary, always check if the item is the `next_expected` one first to bypass buffering entirely, and use `dict.pop` with a sentinel for faster extraction loops.

## 2024-05-27 - [Fast zero-row masking using computed indices]
**Learning:** When needing to check if an entire row is fully zeroed out (e.g. from a padding mask) and the index of the last valid item (`last_token_indices`) is already computed, using direct array indexing `mask[batch_indices, last_token_indices] != 0` is O(1) per row. Using `mask.any(axis=1)` is O(N) per row, introducing unnecessary performance hits on hot paths with very long sequences.
**Action:** When extracting data using index arrays from a sequence, rely on direct boolean checks on those specific indices to determine row validity rather than scanning the entire original boolean mask row.
