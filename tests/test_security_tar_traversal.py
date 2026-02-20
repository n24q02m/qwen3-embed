import os
from unittest.mock import MagicMock, patch

import pytest

from qwen3_embed.common.model_management import ModelManagement


def test_decompress_to_cache_uses_filter_data():
    """
    Verifies that tarfile extraction uses filter='data' to prevent path traversal.
    """
    # Create a dummy tar.gz file path (we need it to exist for os.path.isfile check)
    dummy_tar = "dummy_test.tar.gz"

    # Create an empty file to pass isfile check
    with open(dummy_tar, "wb") as f:
        f.write(b"")

    try:
        # We need to mock tarfile.open to avoid actually opening the invalid file
        with patch("tarfile.open") as mock_open:
            mock_tar = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_tar

            # Call the function
            ModelManagement.decompress_to_cache(dummy_tar, "cache_dir")

            # Check if extractall was called
            assert mock_tar.extractall.called

            # Check if filter='data' was passed
            # We check verify that 'filter' is in kwargs and equals 'data'
            args, kwargs = mock_tar.extractall.call_args

            if "filter" not in kwargs:
                pytest.fail(
                    "tarfile.extractall() called without 'filter' argument. Vulnerable to path traversal!"
                )

            assert kwargs["filter"] == "data", (
                f"extractall called with filter='{kwargs['filter']}', expected 'data'"
            )

    finally:
        if os.path.exists(dummy_tar):
            os.remove(dummy_tar)
