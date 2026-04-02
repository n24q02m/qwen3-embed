import io
import tarfile
import unittest.mock as mock

import pytest

from qwen3_embed.common.model_management import ModelManagement


def test_decompress_symlink_traversal_prevention(tmp_path):
    """Verify that relative symlink traversal is prevented."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    malicious_tar = tmp_path / "malicious_symlink.tar.gz"
    with tarfile.open(malicious_tar, "w:gz") as tar:
        info = tarfile.TarInfo(name="evil_symlink")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside.txt"
        tar.addfile(info)

    def mock_hasattr(obj, name):
        if obj is tarfile and name == "data_filter":
            return False
        return hasattr(obj, name)

    with mock.patch("qwen3_embed.common.model_management.hasattr", side_effect=mock_hasattr):
        with pytest.raises(tarfile.TarError, match="Attempted path traversal via link"):
            ModelManagement.decompress_to_cache(str(malicious_tar), str(cache_dir))

    assert not (cache_dir / "evil_symlink").exists()


def test_decompress_hardlink_traversal_prevention(tmp_path):
    """Verify that relative hardlink traversal is prevented."""
    cache_dir = tmp_path / "cache_hard"
    cache_dir.mkdir()

    malicious_tar = tmp_path / "malicious_hardlink.tar.gz"
    with tarfile.open(malicious_tar, "w:gz") as tar:
        # Create a file to link to
        data = b"target"
        target_info = tarfile.TarInfo(name="target.txt")
        target_info.size = len(data)
        tar.addfile(target_info, io.BytesIO(data))

        info = tarfile.TarInfo(name="evil_hardlink")
        info.type = tarfile.LNKTYPE
        info.linkname = "../outside_hard.txt"
        tar.addfile(info)

    def mock_hasattr(obj, name):
        if obj is tarfile and name == "data_filter":
            return False
        return hasattr(obj, name)

    with mock.patch("qwen3_embed.common.model_management.hasattr", side_effect=mock_hasattr):
        with pytest.raises(tarfile.TarError, match="Attempted path traversal via link"):
            ModelManagement.decompress_to_cache(str(malicious_tar), str(cache_dir))

    assert not (cache_dir / "evil_hardlink").exists()


def test_decompress_absolute_link_traversal_prevention(tmp_path):
    """Verify that absolute link traversal is prevented."""
    cache_dir = tmp_path / "cache_abs"
    cache_dir.mkdir()

    malicious_tar = tmp_path / "malicious_abs_link.tar.gz"
    with tarfile.open(malicious_tar, "w:gz") as tar:
        info = tarfile.TarInfo(name="evil_abs_symlink")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    def mock_hasattr(obj, name):
        if obj is tarfile and name == "data_filter":
            return False
        return hasattr(obj, name)

    with mock.patch("qwen3_embed.common.model_management.hasattr", side_effect=mock_hasattr):
        with pytest.raises(tarfile.TarError, match="Attempted path traversal via link"):
            ModelManagement.decompress_to_cache(str(malicious_tar), str(cache_dir))


def test_decompress_safe_symlink(tmp_path):
    """Verify that safe internal symlinks are allowed."""
    cache_dir = tmp_path / "cache_safe"
    cache_dir.mkdir()

    safe_tar = tmp_path / "safe_symlink.tar.gz"
    with tarfile.open(safe_tar, "w:gz") as tar:
        data = b"hello"
        info_file = tarfile.TarInfo(name="subdir/hello.txt")
        info_file.size = len(data)
        tar.addfile(info_file, io.BytesIO(data))

        info_link = tarfile.TarInfo(name="link_to_hello")
        info_link.type = tarfile.SYMTYPE
        info_link.linkname = "subdir/hello.txt"
        tar.addfile(info_link)

    def mock_hasattr(obj, name):
        if obj is tarfile and name == "data_filter":
            return False
        return hasattr(obj, name)

    with mock.patch("qwen3_embed.common.model_management.hasattr", side_effect=mock_hasattr):
        ModelManagement.decompress_to_cache(str(safe_tar), str(cache_dir))

    assert (cache_dir / "subdir" / "hello.txt").read_bytes() == b"hello"
    assert (cache_dir / "link_to_hello").exists()
