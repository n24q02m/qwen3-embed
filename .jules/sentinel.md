## 2025-02-23 - Insecure Temporary Directory Permissions
**Vulnerability:** The `mkdir` function in `qwen3_embed/common/model_management.py` created a cache temporary directory `cache_tmp_dir` using the default process umask. On multi-user systems, this could expose downloaded and decompressed model files to unauthorized users before they were moved to their final destination.
**Learning:** Temporary cache directories handling sensitive or large model data require explicit restrictive permissions (e.g., `0o700`) to prevent local unauthorized access, rather than relying on the system's default umask.
**Prevention:** Explicitly use `os.chmod` inside a `contextlib.suppress(OSError)` block after directory creation to robustly apply `0o700` permissions without failing on unsupported platforms.
