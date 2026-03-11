## 2024-05-24 - Tar Slip Vulnerability Prevention

**Vulnerability:** Arbitrary File Write via Archive Extraction (Tar Slip)
When using `tarfile.extractall()` without validating the extracted paths, an attacker can create a malicious archive containing files with absolute paths or `../` sequences. This allows the attacker to write files outside the intended extraction directory, potentially overwriting critical system files or planting malware.

**Learning:** `tarfile.extractall()` is inherently dangerous if used directly on untrusted input because it does not automatically sanitize member paths. The `filter="data"` argument helps in Python 3.12+, but manual validation is crucial for complete safety and backward compatibility.

**Prevention:** Before extraction, iterate through `tar.getmembers()` and use `os.path.abspath()` to compute the intended absolute extraction path for each member. Compare this computed path to the target directory (`os.path.abspath(target_dir)`). If the computed path does not strictly start with the target directory, raise an exception (e.g., `tarfile.TarError`) to halt the process and prevent arbitrary file writes. Unconditionally clean up the partial extraction directory on failure to avoid leaving unsafe remnants.
