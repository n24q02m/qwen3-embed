import tarfile
import io
import pytest
from pathlib import Path
from qwen3_embed.common.model_management import ModelManagement

@pytest.fixture
def malicious_tar(tmp_path):
    tar_path = tmp_path / "malicious.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        # Create a file using parent directory traversal (../)
        # This attempts to write to a file outside the extraction directory
        t = tarfile.TarInfo("../exploit.txt")
        t.size = len(b"pwned")
        tar.addfile(t, io.BytesIO(b"pwned"))
    return tar_path

def test_tar_slip_vulnerability(tmp_path, malicious_tar):
    """
    Test that extracting a tar file with '../' paths raises an error
    and does not write files outside the destination directory.
    """
    extraction_dir = tmp_path / "extract"
    extraction_dir.mkdir()

    # We expect this to fail securely (raise an exception due to filter='data')
    # It typically raises tarfile.LinkOutsideDestinationError or MemberOutsideDestinationError
    # which are subclasses of TarError, which ModelManagement catches and re-raises as ValueError.
    with pytest.raises((ValueError, tarfile.TarError)):
        ModelManagement.decompress_to_cache(str(malicious_tar), str(extraction_dir))

    # Check if exploit.txt was created in tmp_path (one level up from extraction_dir)
    exploit_file = tmp_path / "exploit.txt"
    assert not exploit_file.exists(), "Security vulnerability: File was extracted outside the target directory!"

def test_tar_absolute_path_safety(tmp_path):
    """
    Test that extracting a tar file with absolute paths is handled safely.
    Standard tarfile behavior strips leading slashes, making them relative.
    The key check is that it does NOT write to the absolute path outside.
    """
    tar_path = tmp_path / "absolute.tar.gz"
    # Create a file with absolute path outside extraction dir
    abs_exploit = tmp_path / "abs_exploit.txt"

    with tarfile.open(tar_path, "w:gz") as tar:
        t = tarfile.TarInfo(str(abs_exploit)) # Absolute path
        t.size = len(b"pwned")
        tar.addfile(t, io.BytesIO(b"pwned"))

    extraction_dir = tmp_path / "extract_abs"
    extraction_dir.mkdir()

    # Should not raise, but should sanitize
    ModelManagement.decompress_to_cache(str(tar_path), str(extraction_dir))

    assert not abs_exploit.exists(), "Security vulnerability: Absolute path extraction succeeded!"

    # Verify it was extracted inside the directory (sanitized)
    # The path will be stripped of leading slash
    stripped_path = str(abs_exploit).lstrip("/")
    safe_path = extraction_dir / stripped_path
    assert safe_path.exists(), "Sanitized file should exist inside extraction directory"
