## 2025-02-18 - [Fix Zip Slip in Model Download]
**Vulnerability:** Found a potential "Zip Slip" vulnerability in `ModelManagement.decompress_to_cache` where `tarfile.extractall` was used without a filter, allowing malicious tar files to write outside the destination directory.
**Learning:** Python's `tarfile` module (pre-3.14/3.12 without filter) defaults to unsafe extraction. Explicitly setting `filter='data'` is crucial for security when handling untrusted archives, even if newer Python versions might default to it.
**Prevention:** Always use `filter='data'` (or stricter) when using `tarfile.extractall`.
