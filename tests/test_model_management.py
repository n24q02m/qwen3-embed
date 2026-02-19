"""Tests for model management utility functions."""

import pytest

from qwen3_embed.common.model_management import ModelManagement


class TestDecompressToCache:
    """Tests for decompress_to_cache method."""

    def test_decompress_nonexistent_file(self, tmp_path):
        """Test that decompress_to_cache raises ValueError for a nonexistent file."""
        nonexistent_file = tmp_path / "nonexistent.tar.gz"
        with pytest.raises(ValueError, match="does not exist or is not a file"):
            ModelManagement.decompress_to_cache(str(nonexistent_file), str(tmp_path))

    def test_decompress_invalid_extension(self, tmp_path):
        """Test that decompress_to_cache raises ValueError for a file with invalid extension."""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.touch()
        with pytest.raises(ValueError, match="is not a .tar.gz file"):
            ModelManagement.decompress_to_cache(str(invalid_file), str(tmp_path))

    def test_decompress_directory(self, tmp_path):
        """Test that decompress_to_cache raises ValueError if the path is a directory."""
        directory = tmp_path / "directory.tar.gz"
        directory.mkdir()
        with pytest.raises(ValueError, match="does not exist or is not a file"):
            ModelManagement.decompress_to_cache(str(directory), str(tmp_path))

    def test_decompress_corrupted_tar_gz(self, tmp_path):
        """Test that decompress_to_cache raises ValueError for a corrupted .tar.gz file."""
        corrupted_file = tmp_path / "corrupted.tar.gz"
        corrupted_file.write_text("not a tar file")

        cache_dir = tmp_path / "cache_tmp"
        cache_dir.mkdir()

        with pytest.raises(ValueError, match="An error occurred while decompressing"):
            ModelManagement.decompress_to_cache(str(corrupted_file), str(cache_dir))
