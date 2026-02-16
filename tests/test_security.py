import tarfile

import pytest

from qwen3_embed.common.model_management import ModelManagement


class TestSecurity:
    def test_zip_slip_vulnerability(self, tmp_path):
        """
        Test that ModelManagement.decompress_to_cache prevents Zip Slip attacks.

        This test creates a malicious tarball with a file path containing '../'
        and verifies that extraction is blocked (raises an error) or at least
        does not write outside the destination when the fix is applied.
        """
        # Create a malicious tarball
        tar_path = tmp_path / "malicious.tar.gz"
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        # The file we want to write outside the target directory
        exploit_file_name = "exploit.txt"

        # Create the tarball
        with tarfile.open(tar_path, "w:gz") as tar:
            # Create a file info object
            tarinfo = tarfile.TarInfo(name=f"../{exploit_file_name}")
            tarinfo.size = len(b"exploit")

            # Add it to the tarball
            import io

            tar.addfile(tarinfo, io.BytesIO(b"exploit"))

        # Attempt to decompress
        # With filter='data', this MUST raise a ValueError wrapping tarfile.FilterError (or LinkOutsideDestinationError)
        # Note: ModelManagement wraps TarError in ValueError with message containing the original error

        with pytest.raises(ValueError, match="outside the destination"):
            ModelManagement.decompress_to_cache(str(tar_path), str(target_dir))

        # Check if the file was written outside
        exploit_path = tmp_path / exploit_file_name

        # Ensure it was NOT written
        assert not exploit_path.exists(), (
            "Zip Slip vulnerability detected: File written outside target directory!"
        )
