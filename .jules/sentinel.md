## 2025-05-13 - Path Traversal in Tarfile Extraction
**Vulnerability:** `tarfile.extractall` without a filter allows path traversal attacks (e.g., `../../etc/passwd`) in Python versions prior to 3.14 (defaulting to 'fully_trusted' in <3.12, or effectively insecure in 3.12/3.13 without explicit filter).
**Learning:** Python's standard library functions for archive extraction are not secure by default against malicious archives until Python 3.14. Explicitly using `filter='data'` (available since 3.12) is necessary to prevent overwriting arbitrary files.
**Prevention:** Always use `filter='data'` when extracting tar archives in Python 3.12+ (and backports) to enforce secure extraction policies.
