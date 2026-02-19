from unittest.mock import MagicMock, patch

from qwen3_embed.common.model_management import ModelManagement


class TestModelManagement:
    @patch("qwen3_embed.common.model_management.tarfile.open")
    @patch("qwen3_embed.common.model_management.os.path.isfile")
    def test_decompress_to_cache_uses_safe_filter(self, mock_isfile, mock_tarfile_open):
        """Test that decompress_to_cache uses filter='data' for safe extraction."""
        # Setup mocks
        mock_isfile.return_value = True

        mock_tar = MagicMock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar

        targz_path = "/tmp/test_model.tar.gz"
        cache_dir = "/tmp/cache_dir"

        # Call the method
        ModelManagement.decompress_to_cache(targz_path, cache_dir)

        # Verify tarfile.open was called correctly
        mock_tarfile_open.assert_called_once_with(targz_path, "r:gz")

        # Verify extractall was called with filter='data'
        # This assertion is expected to fail before the fix
        mock_tar.extractall.assert_called_once_with(path=cache_dir, filter="data")
