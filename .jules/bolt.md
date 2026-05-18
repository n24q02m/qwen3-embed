## 2026-05-01 - [Fast 2-class classification in Numpy]
**Learning:** For 2-class classification (binary softmax) in numpy, computing the sigmoid on the difference of logits is fast. This can be further optimized by using `np.subtract(yes_no_logits[:, 0], yes_no_logits[:, 1], dtype=np.float32)` instead of casting individually to floats and then subtracting, saving array allocations and execution time. Also, using in-place operations like `np.exp(diff, out=diff)` and `np.reciprocal(diff, out=diff)` avoids extra memory overhead.
**Action:** Always prefer `np.subtract` over `array.astype() - array.astype()` and utilize in-place mutation where possible when optimizing hot paths involving numpy arrays.

## 2024-05-24 - [Fast iterable chunking with walrus operator]
**Learning:** When chunking generic iterables in hot paths using `itertools.islice`, using the walrus operator (`while b := list(islice(source_iter, size)):`) instead of a `while True` loop with an explicit length check reduces bytecode execution overhead.
**Action:** Use the walrus operator for iterator exhaustion loops to avoid unnecessary length checks and loop breaks.

## 2024-05-06 - [Fast single scalar math operations]
**Learning:** For mathematical operations on single scalar values (e.g., computing a sigmoid from a logit difference), using Python's built-in `math.exp` is significantly faster than `numpy.exp` due to the avoidance of numpy's C-API dispatch and object creation overhead. Because `math.exp` raises an `OverflowError` for large negative exponents, it should be wrapped in a `try...except OverflowError` block to handle edge cases appropriately (e.g., returning 0.0 or 1.0 for sigmoid boundaries).
**Action:** Use `math.exp` instead of `np.exp` when operating on single scalar values, wrapping it in a `try...except OverflowError` block to handle numerical boundaries.

## 2025-05-01 - [Fast last token index calculation]
**Learning:** When calculating the last token index for strictly right-padded attention masks (which consist of contiguous 1s followed by 0s) in numpy, using `mask.sum(axis=1) - 1` is significantly (~5x) faster than `seq_len - 1 - np.argmax(mask[:, ::-1], axis=1)` because it avoids array reversal and argmax allocation overhead.
**Action:** Always prefer `mask.sum(axis=1) - 1` over `argmax` combined with array reversal when finding the last token index for right-padded masks.
