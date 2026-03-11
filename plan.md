1. **Analyze `download_file_from_gcs` in `qwen3_embed/common/model_management.py`**
   - The function downloads files but does not verify the content against an expected hash.
   - We need to extract the expected MD5 hash from the `x-goog-hash` HTTP response header, which typically looks like `x-goog-hash: crc32c=..., md5=...`.
   - The MD5 string in `x-goog-hash` is base64 encoded. We will decode it and convert it to a hex string for comparison.
   - We will compute the MD5 hash of the downloaded content dynamically using `hashlib.md5()`.
   - If the downloaded file's hash does not match the expected hash, we raise a `ValueError` and remove the partially downloaded/corrupted file.

2. **Modify `qwen3_embed/common/model_management.py`**
   - Import `base64` and `hashlib`.
   - Parse `x-goog-hash` header:
     ```python
        expected_md5 = None
        x_goog_hash = response.headers.get("x-goog-hash")
        if x_goog_hash:
            for h in x_goog_hash.split(","):
                h = h.strip()
                if h.startswith("md5="):
                    expected_md5 = base64.b64decode(h[4:]).hex()
     ```
   - Initialize `md5_hash = hashlib.md5()`.
   - In the download loop, update the hash: `md5_hash.update(chunk)`.
   - After the loop, verify:
     ```python
        if expected_md5 and md5_hash.hexdigest() != expected_md5:
            os.remove(output_path)
            raise ValueError(
                f"MD5 checksum mismatch for downloaded file. Expected {expected_md5}, got {md5_hash.hexdigest()}"
            )
     ```
   - Ensure `os.remove(output_path)` runs correctly.

3. **Add Tests to `tests/test_model_management.py`**
   - Mock a successful download with a matching `x-goog-hash`.
   - Mock a corrupted download with a mismatched `x-goog-hash`, expecting a `ValueError` and ensuring `output_path` is deleted.
   - Mock a download without an `x-goog-hash` (should pass normally).

4. **Verify Tests and Linting**
   - Run `uv run pytest tests/test_model_management.py`
   - Run `uv run ruff check .`

5. **Complete Pre-Commit Steps**
   - Follow instructions from `pre_commit_instructions`

6. **Submit PR**
   - Submit the fix with an appropriate commit message and title.
