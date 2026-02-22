import tarfile
import io
import pytest
from pathlib import Path
from qwen3_embed.common.model_management import ModelManagement

def create_malicious_tar(filename: str):
    with tarfile.open(filename, "w:gz") as tar:
        # Create a file that will be written outside the target directory
        # This simulates a "Zip Slip" / "Tar Slip" attack
        t = tarfile.TarInfo("../pwned.txt")
        t.size = len(b"hacked")
        tar.addfile(t, io.BytesIO(b"hacked"))

def test_tar_slip_vulnerability(tmp_path):
    tar_path = tmp_path / "evil.tar.gz"
    extract_dir = tmp_path / "extract_dir"
    target_file = tmp_path / "pwned.txt"

    extract_dir.mkdir()
    create_malicious_tar(str(tar_path))

    # With the vulnerability present, this call succeeds and writes the file.
    # With the fix, it should raise ValueError (wrapping TarError).

    try:
        ModelManagement.decompress_to_cache(str(tar_path), str(extract_dir))
    except ValueError as e:
        # Expected after fix
        assert "outside the destination" in str(e)
        return

    # If we are here, no exception was raised.
    # Check if vulnerability was exploited
    if target_file.exists():
        pytest.fail("Vulnerability exploited: File created outside extraction directory!")

    # If file not created and no exception, it's weird but maybe safe?
    # But since we want to enforce 'data' filter which raises exception, we should fail here too.
    pytest.fail("Extraction succeeded unexpectedly without raising security exception.")
