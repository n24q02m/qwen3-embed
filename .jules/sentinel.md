## 2024-05-18 - [Fix Missing verify=True in requests.get]
**Vulnerability:** The code `requests.get` implicitly relies on the default configuration for TLS verification. This can be maliciously or accidentally disabled via the `REQUESTS_CA_BUNDLE` environment variable, leading to Man-In-The-Middle (MITM) attacks when downloading models from untrusted or tampered sources.
**Learning:** Explicit configuration (such as `verify=True` in HTTP requests) provides defense-in-depth against environment manipulation.
**Prevention:** Always explicitly set `verify=True` when executing HTTP requests to external or potentially unauthenticated sources using the `requests` library.
## 2024-05-18 - [Fix Path Traversal in Tarfile Symlinks]
**Vulnerability:** The `decompress_to_cache` method validated `member.name` for path traversal but failed to validate `member.linkname` for symlinks and hardlinks. This could allow an attacker to create a symlink pointing outside the target directory and subsequently overwrite arbitrary files if another archive member extracts to that symlink.
**Learning:** When manually validating archive members to prevent path traversal (e.g., in a `tarfile.extractall` fallback), you must validate the destination of symlinks and hardlinks (`member.linkname`) in addition to `member.name`.
**Prevention:** Always reject absolute paths or traversals in `member.linkname` when `member.issym()` or `member.islnk()` is true.
