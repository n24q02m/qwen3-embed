## 2026-02-28 - [Path Traversal in `tarfile.extractall`]
**Vulnerability:** The `tarfile.extractall` method in Python prior to 3.14 defaults to a vulnerable `filter="fully_trusted"`, making it susceptible to Zip Slip / path traversal attacks if extracting an untrusted `.tar.gz`.
**Learning:** Even though the environment is restricted to >=3.13, explicitly passing `filter="data"` is still needed because the default doesn't become secure until Python 3.14.
**Prevention:** Always explicitly pass `filter="data"` to `tarfile.extractall` in Python 3.12+ code to block path traversal vulnerabilities.
