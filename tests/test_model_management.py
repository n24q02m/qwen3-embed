"""Unit tests for ModelManagement.decompress_to_cache."""

import os
import shutil
import tarfile
import tempfile
import unittest

from qwen3_embed.common.model_management import ModelManagement


class TestModelManagementDecompressToCache(unittest.TestCase):
    """Test suite for ModelManagement.decompress_to_cache."""

    def setUp(self):
        """Set up temporary directories."""
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.test_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir)

    def create_dummy_tar_gz(self, file_path, contents=None):
        """Helper to create a dummy .tar.gz file."""
        if contents is None:
            contents = {"test_file.txt": b"Hello World"}

        # Use temp file approach as it is reliable
        with tarfile.open(file_path, "w:gz") as tar:
            for name, data in contents.items():
                t = os.path.join(self.test_dir, "temp_content")
                with open(t, "wb") as f:
                    f.write(data)
                tar.add(t, arcname=name)
                os.remove(t)

    def test_decompress_to_cache_valid(self):
        """Test decompression of a valid .tar.gz file."""
        tar_path = os.path.join(self.test_dir, "valid.tar.gz")
        self.create_dummy_tar_gz(
            tar_path, {"file1.txt": b"content1", "subdir/file2.txt": b"content2"}
        )

        target_dir = os.path.join(self.cache_dir, "target_valid")
        os.makedirs(target_dir, exist_ok=True)

        ModelManagement.decompress_to_cache(tar_path, target_dir)

        self.assertTrue(os.path.exists(os.path.join(target_dir, "file1.txt")))
        with open(os.path.join(target_dir, "file1.txt"), "rb") as f:
            self.assertEqual(f.read(), b"content1")

        self.assertTrue(os.path.exists(os.path.join(target_dir, "subdir", "file2.txt")))
        with open(os.path.join(target_dir, "subdir", "file2.txt"), "rb") as f:
            self.assertEqual(f.read(), b"content2")

    def test_decompress_to_cache_invalid_file_path(self):
        """Test with a non-existent file path."""
        with self.assertRaises(ValueError) as cm:
            ModelManagement.decompress_to_cache("non_existent.tar.gz", self.cache_dir)
        self.assertIn("does not exist", str(cm.exception))

    def test_decompress_to_cache_invalid_extension(self):
        """Test with an invalid file extension."""
        invalid_path = os.path.join(self.test_dir, "invalid.zip")
        with open(invalid_path, "w") as f:
            f.write("dummy")

        with self.assertRaises(ValueError) as cm:
            ModelManagement.decompress_to_cache(invalid_path, self.cache_dir)
        self.assertIn("is not a .tar.gz file", str(cm.exception))

    def test_decompress_to_cache_corrupted_file(self):
        """Test with a corrupted .tar.gz file."""
        corrupted_path = os.path.join(self.test_dir, "corrupted.tar.gz")
        with open(corrupted_path, "wb") as f:
            f.write(b"not a tar file")

        target_dir = os.path.join(
            self.cache_dir, "tmp_corrupted"
        )  # Must have "tmp" in name to trigger cleanup
        os.makedirs(target_dir, exist_ok=True)

        with self.assertRaises(ValueError) as cm:
            ModelManagement.decompress_to_cache(corrupted_path, target_dir)

        self.assertIn("An error occurred while decompressing", str(cm.exception))
        # Verify cleanup happened
        self.assertFalse(os.path.exists(target_dir))

    def test_decompress_to_cache_path_traversal(self):
        """Test that path traversal attempts raise an error (Tar Slip)."""
        tar_path = os.path.join(self.test_dir, "slip.tar.gz")

        # Create a malicious tar file
        dummy_content = os.path.join(self.test_dir, "dummy")
        with open(dummy_content, "w") as f:
            f.write("malicious")

        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(dummy_content, arcname="../outside_cache.txt")

        target_dir = os.path.join(self.cache_dir, "target_slip")
        os.makedirs(target_dir, exist_ok=True)

        try:
            ModelManagement.decompress_to_cache(tar_path, target_dir)
        except ValueError:
            # Expected failure if filter='data' is enabled and working
            pass
        else:
            # If no exception, verify if file was written outside
            outside_path = os.path.join(self.cache_dir, "outside_cache.txt")
            if os.path.exists(outside_path):
                self.fail("Path traversal vulnerability: File written outside target directory!")
            else:
                # It might have been filtered silently or failed for other reasons
                pass


if __name__ == "__main__":
    unittest.main()
