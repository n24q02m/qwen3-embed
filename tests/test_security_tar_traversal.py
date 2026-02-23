import tarfile
from unittest.mock import MagicMock, patch

import pytest

from qwen3_embed.common.model_management import ModelManagement


class TestTarSecurity:
    @pytest.fixture
    def malicious_tar(self, tmp_path):
        """Creates a malicious tar file with a path traversal payload."""
        tar_path = tmp_path / "malicious.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Create a dummy file
            dummy_file = tmp_path / "evil.txt"
            dummy_file.write_text("You have been pwned!")

            # Add it to archive with a path traversal name
            tar.add(dummy_file, arcname="../evil_extracted.txt")
        return tar_path

    def test_tar_slip_prevention(self, malicious_tar, tmp_path):
        """
        Verifies that extracting a malicious tar file raises an error.
        Note: This might pass even without the fix on Python 3.12+ due to safer defaults,
        but we want to ensure it fails securely.
        """
        extract_dir = tmp_path / "extraction_target"
        extract_dir.mkdir()

        with pytest.raises(ValueError, match="An error occurred while decompressing"):
            ModelManagement.decompress_to_cache(str(malicious_tar), str(extract_dir))

        # Verify file was NOT extracted outside
        evil_path = extract_dir.parent / "evil_extracted.txt"
        assert not evil_path.exists(), "File was extracted outside the target directory!"

    def test_explicit_filter_data(self, malicious_tar, tmp_path):
        """
        Verifies that tarfile.extractall is called with filter='data'.
        This ensures explicit security regardless of Python version defaults.
        """
        extract_dir = tmp_path / "extraction_target"
        extract_dir.mkdir(exist_ok=True)

        # We mock tarfile.open to return a mock tar object so we can inspect calls
        with patch("tarfile.open") as mock_open:
            mock_tar = MagicMock()
            # Configure context manager behavior
            mock_open.return_value.__enter__.return_value = mock_tar

            ModelManagement.decompress_to_cache(str(malicious_tar), str(extract_dir))

            # Assert extractall was called with filter='data'
            mock_tar.extractall.assert_called_once()
            call_kwargs = mock_tar.extractall.call_args.kwargs

            # Check if filter argument is present and correct
            if "filter" not in call_kwargs:
                pytest.fail(
                    "extractall called without 'filter' argument - prone to Tar Slip vulnerability!"
                )

            assert call_kwargs["filter"] == "data", (
                f"extractall called with filter='{call_kwargs['filter']}', expected 'data'"
            )
