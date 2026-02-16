# Sentinel's Journal

## 2026-02-16 - Zip Slip in Model Decompression
**Vulnerability:** Zip Slip (path traversal) vulnerability in `tarfile.extractall()` within `ModelManagement.decompress_to_cache`.
**Learning:** `tarfile.extractall()` defaults to trusted extraction (no filter) in older Python versions (and even 3.12 without explicit filter argument), allowing malicious tarballs to overwrite files outside the target directory.
**Prevention:** Always use `filter='data'` (or 'tar' if permissions are needed, but 'data' is safer for most cases) in `tarfile.extractall()` to block path traversal.
