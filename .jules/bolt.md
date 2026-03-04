## 2024-03-04 - Function memoization in get_all_punctuation
**Learning:** `get_all_punctuation` iterates over `sys.maxunicode` to compute a set of all punctuation characters. This computation is expensive (O(N) operations, N=1,114,111) and yields the exact same static result on every call. It's a textbook case for memoization.
**Action:** Use `@functools.lru_cache()` decorator to memoize the static output, ensuring the O(N) calculation only occurs once per process execution.
