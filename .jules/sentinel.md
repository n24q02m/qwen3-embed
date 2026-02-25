# Sentinel's Journal

## 2025-05-20 - Zip Slip in Model Decompression
**Vulnerability:** A "Zip Slip" (Path Traversal) vulnerability was found in `ModelManagement.decompress_to_cache` which uses `tarfile.extractall` without a `filter`.
**Learning:** Python's `tarfile.extractall` is unsafe by default in versions prior to 3.14 (including 3.13, which this project supports), as it allows extracting files outside the target directory if the archive contains members with absolute paths or `..` components.
**Prevention:** Always explicitly set `filter='data'` (or `tarfile.data_filter` in older Python versions via compatibility shim if needed, but `filter='data'` is available in 3.12+) when using `extractall`. This mitigates the risk by enforcing that extracted paths remain within the destination.
