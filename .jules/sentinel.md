## 2024-10-24 - [Atomic File Downloads]
**Vulnerability:** File downloads via GCS were written directly to their final `output_path`. If the process crashed, was interrupted, or the integrity check failed *after* partial writing, a corrupted or malicious file could be left at the expected location, leading to cache poisoning or execution errors in concurrent processes.
**Learning:** `os.replace` is atomic on POSIX systems when the source and destination are on the same filesystem. Always download to a temporary file (`.tmp`) and rename it only *after* complete validation.
**Prevention:** Implement atomic file writes by downloading to an intermediate path and using `os.replace`. Use `contextlib.suppress(OSError)` in a `finally` block to safely clean up the temporary file if the operation aborts.
## 2024-10-24 - [SSRF Bypass in URL Validation]
**Vulnerability:** The `download_file_from_gcs` method validated URLs using `url.startswith('https://storage.googleapis.com/')`. This could be bypassed using credentials syntax (e.g., `https://storage.googleapis.com@127.0.0.1/`), allowing an attacker to fetch arbitrary internal resources (Server-Side Request Forgery).
**Learning:** String matching like `startswith` is insufficient for URL validation because parsers handle auth components differently.
**Prevention:** Always parse URLs using `urllib.parse.urlparse` and validate the `scheme` and `netloc` (or `hostname`) independently.
## 2024-10-24 - [Decompression Bomb Vulnerability]
**Vulnerability:** `decompress_to_cache` extracts `.tar.gz` files from GCS without validating the total uncompressed size, making the system vulnerable to Decompression Bomb (Tar Bomb) attacks.
**Learning:** Always track the total extracted file size across all members during `tar.getmembers()` iteration and raise an error if it exceeds a safe limit (e.g., 20GB).
**Prevention:** Keep track of `total_extracted_size` and check against `max_total_size` during archive member iteration.
