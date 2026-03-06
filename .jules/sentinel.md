## 2025-01-01 - Suppress Bandit B615 for local-only Hugging Face downloads
**Vulnerability:** Unsafe Hugging Face Hub download without revision pinning in snapshot_download() when local_files_only=True.
**Learning:** Bandit raises B615 when snapshot_download is called without a revision pin. However, when local_files_only=True, no remote fetching occurs, so it is safe to suppress this warning.
**Prevention:** Suppress Bandit warnings (e.g., # nosec B615) only when actions are guaranteed to be safe, such as calling snapshot_download with local_files_only=True where no remote fetching occurs. Otherwise, always use explicit revision pinning (e.g., revision=repo_revision) to prevent supply chain attacks or unexpected model tampering.
