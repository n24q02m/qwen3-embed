## 2024-03-01 - Add tests for Tar Slip vulnerability mitigation
**Vulnerability:** Arbitrary file write via archive extraction (`tar.extractall`) without specifying a restrictive `filter` allows path traversal attacks.
**Learning:** Python 3.12+ (and the project's supported >= 3.13) provides a `filter="data"` argument for `tarfile.extractall` that prevents path traversal and block device creation.
**Prevention:** Always use `filter="data"` or `filter="tar"` when using `tarfile.extractall` to ensure secure extraction of untrusted archive files. Verify the behavior using tests that attempt directory traversal.
