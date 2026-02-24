import os
import shutil
import tarfile
import tempfile
import unittest

from qwen3_embed.common.model_management import ModelManagement


class TestModelManagementDecompress(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.test_dir, "cache")
        os.makedirs(self.cache_dir)

        # Create a dummy file content
        self.dummy_content = b"This is a test file."
        self.dummy_filename = "test_file.txt"

        # Create a valid tar.gz file
        self.valid_tar_path = os.path.join(self.test_dir, "valid.tar.gz")
        with tarfile.open(self.valid_tar_path, "w:gz") as tar:
            t = tarfile.TarInfo(name=self.dummy_filename)
            t.size = len(self.dummy_content)
            import io

            tar.addfile(t, io.BytesIO(self.dummy_content))

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_decompress_valid_tar(self):
        """Test valid tar.gz extraction."""
        output_dir = os.path.join(self.cache_dir, "output_valid")
        os.makedirs(output_dir)

        ModelManagement.decompress_to_cache(self.valid_tar_path, output_dir)

        extracted_file = os.path.join(output_dir, self.dummy_filename)
        self.assertTrue(os.path.exists(extracted_file))
        with open(extracted_file, "rb") as f:
            content = f.read()
        self.assertEqual(content, self.dummy_content)

    def test_invalid_file_path(self):
        """Test with non-existent file."""
        with self.assertRaises(ValueError) as cm:
            ModelManagement.decompress_to_cache("non_existent.tar.gz", self.cache_dir)
        self.assertIn("does not exist", str(cm.exception))

    def test_invalid_extension(self):
        """Test with invalid extension."""
        invalid_ext_path = os.path.join(self.test_dir, "invalid.zip")
        with open(invalid_ext_path, "wb") as f:
            f.write(b"dummy")

        with self.assertRaises(ValueError) as cm:
            ModelManagement.decompress_to_cache(invalid_ext_path, self.cache_dir)
        self.assertIn("is not a .tar.gz file", str(cm.exception))

    def test_corrupted_tar(self):
        """Test with corrupted tar.gz file."""
        corrupted_path = os.path.join(self.test_dir, "corrupted.tar.gz")
        with open(corrupted_path, "wb") as f:
            f.write(b"This is not a tar file content")

        output_dir = os.path.join(
            self.cache_dir, "tmp_corrupted"
        )  # Use 'tmp' in name to trigger cleanup
        os.makedirs(output_dir)

        with self.assertRaises(ValueError) as cm:
            ModelManagement.decompress_to_cache(corrupted_path, output_dir)

        # Check if cleanup occurred (directory should be removed)
        self.assertFalse(os.path.exists(output_dir))
        self.assertIn("An error occurred while decompressing", str(cm.exception))

    def test_path_traversal(self):
        """Test path traversal (tar slip) prevention."""
        traversal_tar_path = os.path.join(self.test_dir, "traversal.tar.gz")

        # Create a malicious tar file with '..' entry
        with tarfile.open(traversal_tar_path, "w:gz") as tar:
            t = tarfile.TarInfo(name="../outside.txt")
            t.size = len(self.dummy_content)
            import io

            tar.addfile(t, io.BytesIO(self.dummy_content))

        output_dir = os.path.join(self.cache_dir, "output_traversal")
        os.makedirs(output_dir)

        # This should raise ValueError because of filter='data' (which raises FilterError wrapped in ValueError)
        # If filter='data' is missing, it might succeed (fail to raise) or raise depending on system default.
        # We expect it to raise if implemented correctly.
        try:
            ModelManagement.decompress_to_cache(traversal_tar_path, output_dir)
            # If we reach here, check if file was extracted outside
            outside_file = os.path.join(self.cache_dir, "outside.txt")
            if os.path.exists(outside_file):
                self.fail("Path traversal vulnerability: file extracted outside target directory")
            else:
                # If it didn't extract outside, maybe system default prevented it?
                # But we want to ensure it raises an error.
                pass
        except ValueError as e:
            print(f"Caught expected ValueError: {e}")
            # Check if it's the expected error type for filter='data'
            # Typically "LinkOutsideDestinationError" or similar, wrapped in ValueError.
            # But the code just wraps any TarError.
            # We want to ensure we caught it.
            return

        # If no exception raised, fail
        # Note: On some systems without filter='data', this might pass if the tar extraction is allowed but silently ignored?
        # But tar.extractall usually extracts unless filtered.
        # So we expect failure here if filter='data' is missing.
        self.fail("Path traversal attempt did not raise ValueError")
