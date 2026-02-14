## 2025-02-18 - Zip Slip Vulnerability in Model Download
**Vulnerability:** Found a Zip Slip vulnerability in `decompress_to_cache` where `tarfile.extractall` was used without a filter, allowing malicious tar files to write outside the destination directory.
**Learning:** Even when downloading models from trusted sources (like GCS or HF), the extraction process should be robust against malicious inputs. Python's `tarfile` module before 3.14 (without filter) is unsafe by default.
**Prevention:** Always use `filter='data'` (or safe default in future Python versions) when extracting tar archives.
