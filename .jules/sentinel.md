## 2024-05-15 - Tar Slip Vulnerability in Model Decompression
**Vulnerability:** The `ModelManagement.decompress_to_cache` method uses `tarfile.extractall()` without specifying a filter, allowing path traversal attacks via maliciously crafted `.tar.gz` files (Tar Slip). An attacker could overwrite files outside the extraction directory.
**Learning:** Python's `tarfile` module historically defaulted to an unsafe extraction mode. While newer Python versions introduce `filter='data'`, it must be explicitly opted into until Python 3.14.
**Prevention:** Always use `filter='data'` (or a custom safe filter) when extracting archives with `tarfile` in Python < 3.14.
