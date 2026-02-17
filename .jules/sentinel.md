## 2025-02-18 - Prevent Zip Slip in Tar Extraction
**Vulnerability:** `tarfile.extractall()` was called without a `filter` argument, which can allow malicious tar archives to write files outside the destination directory (Zip Slip).
**Learning:** Python 3.12+ introduced `filter='data'` (and 'tar') to safely extract archives. This must be used whenever `tarfile.extractall()` is called.
**Prevention:** Always use `filter='data'` (or stricter) in `tarfile.extractall()`.
