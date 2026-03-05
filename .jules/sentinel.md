## 2024-05-14 - Unsafe Hugging Face Hub download without revision pinning
**Vulnerability:** Hugging Face model files were downloaded via `snapshot_download` without specifying a `revision`.
**Learning:** `snapshot_download` defaults to the main branch. Without pinning the revision to a specific commit hash, a bad actor could push a malicious file to the model repository, leading to a supply chain attack. We already fetch the `repo_revision` hash but didn't pass it into the remote fetch call.
**Prevention:** When downloading models from Hugging Face using `snapshot_download`, always use explicit revision pinning (e.g., `revision=repo_revision`) to prevent supply chain attacks or unexpected model tampering (fixes Bandit B615).
