## 2026-05-01 - [Fast 2-class classification in Numpy]
**Learning:** For 2-class classification (binary softmax) in numpy, computing the sigmoid on the difference of logits is fast. This can be further optimized by using `np.subtract(yes_no_logits[:, 0], yes_no_logits[:, 1], dtype=np.float32)` instead of casting individually to floats and then subtracting, saving array allocations and execution time. Also, using in-place operations like `np.exp(diff, out=diff)` and `np.reciprocal(diff, out=diff)` avoids extra memory overhead.
**Action:** Always prefer `np.subtract` over `array.astype() - array.astype()` and utilize in-place mutation where possible when optimizing hot paths involving numpy arrays.

## $(date +%Y-%m-%d) - [Fast iterative batching with Walrus Operator]
**Learning:** When chunking generic iterables in hot paths using `itertools.islice`, evaluating truthiness of the resulting list directly via the assignment expression (walrus operator `:=`) avoids an explicit `len() == 0` check.
**Action:** Use `while b := list(islice(source_iter, size)):` instead of `while True:` with `if len(b) == 0: break`. This safely and elegantly reduces bytecode overhead for small batch iteration.
