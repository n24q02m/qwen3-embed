import io
import tarfile

import pytest

from qwen3_embed.common.model_management import ModelManagement


def test_decompress_to_cache_with_malicious_tar_raises_value_error(tmp_path):
    """
    Test that tar files with path traversal attempts are rejected.
    """
    # Create directories
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    target_dir = base_dir / "target"
    target_dir.mkdir()

    # Create a malicious tar file
    tar_path = base_dir / "malicious.tar.gz"

    with tarfile.open(tar_path, "w:gz") as tar:
        # Create a file with path traversal '../evil.txt'
        # This attempts to write to base_dir/evil.txt when extracting to target_dir
        t = tarfile.TarInfo("../evil.txt")
        t.size = len(b"evil content")
        tar.addfile(t, io.BytesIO(b"evil content"))

        # Add a legitimate file too
        t_safe = tarfile.TarInfo("safe.txt")
        t_safe.size = len(b"safe content")
        tar.addfile(t_safe, io.BytesIO(b"safe content"))

    # Attempt to extract using the vulnerable function
    # We expect this to RAISE a ValueError (wrapping the underlying security error)

    try:
        ModelManagement.decompress_to_cache(str(tar_path), str(target_dir))
        pytest.fail("Should have raised ValueError due to path traversal attempt")
    except ValueError:
        # Success: The operation was blocked.
        pass
    except Exception as e:
        pytest.fail(f"Raised unexpected exception type: {type(e)}")

    # Verify that the malicious file was NOT created outside target_dir
    evil_path = base_dir / "evil.txt"
    assert not evil_path.exists(), (
        "Vulnerability exploited: File written outside target directory!"
    )


def test_decompress_to_cache_with_valid_tar_extracts_correctly(tmp_path):
    """
    Test that valid tar files are extracted correctly.
    """
    base_dir = tmp_path / "valid"
    base_dir.mkdir()
    target_dir = base_dir / "target"
    target_dir.mkdir()

    tar_path = base_dir / "valid.tar.gz"

    with tarfile.open(tar_path, "w:gz") as tar:
        t = tarfile.TarInfo("good.txt")
        t.size = len(b"good content")
        tar.addfile(t, io.BytesIO(b"good content"))

    ModelManagement.decompress_to_cache(str(tar_path), str(target_dir))

    assert (target_dir / "good.txt").exists()
    assert (target_dir / "good.txt").read_text() == "good content"


if __name__ == "__main__":
    pytest.main([__file__])
