## 2024-10-24 - [Atomic File Downloads]
**Vulnerability:** File downloads via GCS were written directly to their final `output_path`. If the process crashed, was interrupted, or the integrity check failed *after* partial writing, a corrupted or malicious file could be left at the expected location, leading to cache poisoning or execution errors in concurrent processes.
**Learning:** `os.replace` is atomic on POSIX systems when the source and destination are on the same filesystem. Always download to a temporary file (`.tmp`) and rename it only *after* complete validation.
**Prevention:** Implement atomic file writes by downloading to an intermediate path and using `os.replace`. Use `contextlib.suppress(OSError)` in a `finally` block to safely clean up the temporary file if the operation aborts.
## 2024-10-24 - [SSRF Bypass in URL Validation]
**Vulnerability:** The `download_file_from_gcs` method validated URLs using `url.startswith('https://storage.googleapis.com/')`. This could be bypassed using credentials syntax (e.g., `https://storage.googleapis.com@127.0.0.1/`), allowing an attacker to fetch arbitrary internal resources (Server-Side Request Forgery).
**Learning:** String matching like `startswith` is insufficient for URL validation because parsers handle auth components differently.
**Prevention:** Always parse URLs using `urllib.parse.urlparse` and validate the `scheme` and `netloc` (or `hostname`) independently.

## 2026-05-05 - [Decompression Bomb Vulnerability]
**Vulnerability:** The `decompress_to_cache` method did not restrict the total uncompressed size of a tar archive. A malicious tar file (tar bomb) could consume excessive disk space or memory.
**Learning:** Extracted tar members can expand to gigabytes or terabytes from a small archive, resulting in Resource Exhaustion (DoS).
**Prevention:** Track the running total of `.size` properties from `tar.getmembers()` and raise an exception if it exceeds a maximum safe limit (e.g., 20 GB).

## 2024-05-30 - Fix arbitrary permission modification via symlink attacks
**Vulnerability:** os.chmod and Path.chmod follow symlinks by default. When setting permissions on a directory that an attacker can preemptively create as a symlink (e.g. cache directory), the permissions of the symlink target (such as /etc/passwd) are altered, causing Local Privilege Escalation or DoS.
**Learning:** Always use `follow_symlinks=False` when calling `os.chmod` on dynamically created directories that might reside in shared or user-controlled locations.
**Prevention:** Use `os.chmod(path, mode, follow_symlinks=False)` instead of `Path.chmod(mode)`. Since some platforms do not support `follow_symlinks=False` in `os.chmod`, wrap it in `contextlib.suppress(OSError, NotImplementedError)`.
