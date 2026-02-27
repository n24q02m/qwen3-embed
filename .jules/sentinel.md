## 2024-05-14 - Fix Path Traversal in tarfile extraction
**Vulnerability:** Path traversal vulnerability in `tarfile.extractall()` when processing untrusted `.tar.gz` model archives.
**Learning:** Python 3.13 (and prior) uses the 'fully_trusted' extraction filter by default for backwards compatibility, allowing archives with absolute paths or `../` to overwrite arbitrary files on the system.
**Prevention:** Always specify `filter='data'` in `tar.extractall()` when extracting untrusted archives, which strips potentially dangerous features and enforces path restrictions.
