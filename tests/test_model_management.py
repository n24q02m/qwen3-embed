import tarfile
import pytest
import os
from pathlib import Path
from qwen3_embed.common.model_management import ModelManagement

def test_zip_slip_prevention(tmp_path):
    """
    Test that a malicious tarball (Zip Slip) raises an error and does not write outside the cache directory.
    """
    # Create a malicious tar file
    malicious_tar = tmp_path / "malicious.tar.gz"
    target_file = "hacked.txt"

    # We want to try to write to the parent of cache_dir, which is tmp_path
    # So we use "../hacked.txt" relative to cache_dir

    with tarfile.open(malicious_tar, "w:gz") as tar:
        t = tarfile.TarInfo(name=f"../{target_file}")
        t.size = 0
        tar.addfile(t)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Without the fix, this might succeed (and write outside cache_dir)
    # With the fix, this should raise a ValueError (wrapping tarfile.FilterError)

    with pytest.raises(ValueError, match="An error occurred while decompressing"):
        ModelManagement.decompress_to_cache(str(malicious_tar), str(cache_dir))

    # Verify the file was NOT created outside (in tmp_path)
    assert not (tmp_path / target_file).exists()

def test_valid_decompression(tmp_path):
    """
    Test that a valid tarball is decompressed correctly.
    """
    valid_tar = tmp_path / "valid.tar.gz"
    filename = "safe.txt"

    with tarfile.open(valid_tar, "w:gz") as tar:
        t = tarfile.TarInfo(name=filename)
        t.size = 0
        tar.addfile(t)

    cache_dir = tmp_path / "cache_valid"
    cache_dir.mkdir()

    ModelManagement.decompress_to_cache(str(valid_tar), str(cache_dir))

    assert (cache_dir / filename).exists()
