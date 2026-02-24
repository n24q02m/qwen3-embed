from unittest.mock import MagicMock, patch

from qwen3_embed.common.model_management import ModelManagement


def test_decompress_to_cache_uses_secure_filter(tmp_path):
    mock_tar = MagicMock()
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_tar

    with patch("tarfile.open", return_value=mock_context):
        # Create a dummy tar.gz file path
        tar_path = tmp_path / "model.tar.gz"
        tar_path.touch()

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

        # Verify extractall was called with filter='data'
        # The current implementation likely calls it without filter or with None
        mock_tar.extractall.assert_called_once()
        _, kwargs = mock_tar.extractall.call_args
        assert kwargs.get("filter") == "data", (
            "tarfile.extractall should use filter='data' for security"
        )
