## 2024-08-13 - Python 3.12 itertools.batched speedup
**Learning:** `itertools.batched` (added in Python 3.12) is significantly faster for batching iterables (~30% faster) compared to manually using `itertools.islice` in a while loop due to being implemented in C and avoiding Python bytecode overhead.
**Action:** When a codebase supports Python 3.12+ but still needs to support older versions, conditionally use `itertools.batched` based on `sys.version_info` while falling back to the older implementation to get performance benefits where possible.
