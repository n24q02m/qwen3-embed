## 2025-02-18 - [Vulnerability] Zip Slip via tarfile.extractall
**Vulnerability:** Found `tarfile.extractall()` used without `filter='data'`, which can allow arbitrary file overwrite via malicious tar archives.
**Learning:** Python < 3.12 (and even 3.12+ without explicit filter) can be vulnerable by default.
**Prevention:** Always use `filter='data'` (or 'tar') when extracting untrusted tar files.
