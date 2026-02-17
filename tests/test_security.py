import os
import tarfile
import unittest
from unittest.mock import MagicMock, patch

from qwen3_embed.common.model_management import ModelManagement


class TestSecurity(unittest.TestCase):
    @patch("tarfile.open")
    @patch("os.path.isfile")
    def test_tar_extract_filter(self, mock_isfile, mock_tarfile_open):
        """Verify that tarfile extraction uses filter='data' to prevent Zip Slip."""
        # Setup mock
        mock_isfile.return_value = True

        mock_tar = MagicMock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar

        # Create a dummy tar.gz file path
        targz_path = "dummy_model.tar.gz"
        cache_dir = "dummy_cache"

        # Call the function
        ModelManagement.decompress_to_cache(targz_path, cache_dir)

        # Verify extractall was called with filter='data'
        # This will fail if filter='data' is not used
        mock_tar.extractall.assert_called_with(path=cache_dir, filter='data')
