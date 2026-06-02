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

## 2026-06-02 - [Fast iterative token removal]
**Learning:** For iterative removal of multiple literal string tokens (e.g., preventing prompt injection in `_sanitize_input`), using a nested `while True` loop that iterates over the tokens and applies `str.replace` is significantly faster (~2x) than using a compiled regular expression's `subn` method.
**Action:** Use `str.replace` in a loop instead of regex `subn` for iteratively stripping known literal tokens.
