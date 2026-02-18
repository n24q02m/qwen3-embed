import tarfile
from unittest.mock import MagicMock, patch

import pytest

from qwen3_embed.common.model_management import ModelManagement


def test_decompress_to_cache_uses_data_filter(tmp_path):
    # Create a dummy tar.gz file
    tar_path = tmp_path / "test.tar.gz"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create a valid tar.gz file for the function to open
    with tarfile.open(tar_path, "w:gz"):
        pass  # Empty tar is fine for this test as we mock extractall

    # Mock tarfile.open to return our mock tar object
    with patch("tarfile.open") as mock_open:
        mock_tar = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_tar

        # Call the function
        ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

        # Verify extractall was called with filter='data'
        # We check strictly that filter='data' is present in kwargs
        # or as a positional argument (though it's usually kwargs)

        calls = mock_tar.extractall.call_args_list
        assert len(calls) == 1
        args, kwargs = calls[0]

        # In python 3.12+, filter is a keyword argument.
        # We want to ensure it is explicitly set to 'data'.
        assert "filter" in kwargs, "filter argument missing in extractall call"
        assert kwargs["filter"] == "data", f"Expected filter='data', got {kwargs.get('filter')}"


if __name__ == "__main__":
    # If run directly, allow manual verification
    import sys

    try:
        pytest.main([__file__])
    except SystemExit as e:
        sys.exit(e.code)
