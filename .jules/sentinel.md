## 2024-10-24 - [Atomic File Downloads]
**Vulnerability:** File downloads via GCS were written directly to their final `output_path`. If the process crashed, was interrupted, or the integrity check failed *after* partial writing, a corrupted or malicious file could be left at the expected location, leading to cache poisoning or execution errors in concurrent processes.
**Learning:** `os.replace` is atomic on POSIX systems when the source and destination are on the same filesystem. Always download to a temporary file (`.tmp`) and rename it only *after* complete validation.
**Prevention:** Implement atomic file writes by downloading to an intermediate path and using `os.replace`. Use `contextlib.suppress(OSError)` in a `finally` block to safely clean up the temporary file if the operation aborts.
## 2024-10-24 - [SSRF Bypass in URL Validation]
**Vulnerability:** The `download_file_from_gcs` method validated URLs using `url.startswith('https://storage.googleapis.com/')`. This could be bypassed using credentials syntax (e.g., `https://storage.googleapis.com@127.0.0.1/`), allowing an attacker to fetch arbitrary internal resources (Server-Side Request Forgery).
**Learning:** String matching like `startswith` is insufficient for URL validation because parsers handle auth components differently.
**Prevention:** Always parse URLs using `urllib.parse.urlparse` and validate the `scheme` and `netloc` (or `hostname`) independently.
## 2024-11-06 - Decompression Bomb / Tar Bomb Vulnerability
**Vulnerability:** The application was extracting .tar.gz archives without limiting the total uncompressed file size, making it vulnerable to tar bombs (decompression bombs) which could exhaust disk space and cause Denial of Service (DoS).
**Learning:** Archive formats can compress massive amounts of repetitive data into tiny files. Extracting them safely requires checking the uncompressed sizes defined in the archive headers before writing data to disk.
**Prevention:** Accumulate the `member.size` from `tar.getmembers()` during extraction and raise an exception if the total size exceeds a safe threshold (e.g., 20GB).
