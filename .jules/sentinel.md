## 2025-02-12 - Tar Slip Prevention & Python Defaults
**Vulnerability:** Path traversal vulnerability ("Tar Slip") in `tarfile.extractall`.
**Learning:** Python 3.14+ (and patched older versions) defaults to secure extraction, which masked the vulnerability in our tests. However, relying on this implicit behavior is fragile and may vary by environment or Python version.
**Prevention:** Always explicitly set `filter='data'` in `tarfile.extractall` to enforce security guarantees regardless of the runtime environment's default settings.
