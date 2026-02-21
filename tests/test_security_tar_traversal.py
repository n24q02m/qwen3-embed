import shutil
import tarfile

import pytest

from qwen3_embed.common.model_management import ModelManagement


class TestSecurityTarTraversal:
    @pytest.fixture
    def setup_malicious_tar(self, tmp_path):
        tar_path = tmp_path / "malicious.tar.gz"
        extract_dir = tmp_path / "extract_dir"
        target_file = tmp_path / "evil.txt"  # File to be overwritten/created outside extract_dir

        # Ensure clean state
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        if target_file.exists():
            target_file.unlink()

        extract_dir.mkdir()

        # Create malicious tar
        with tarfile.open(tar_path, "w:gz") as tar:
            # Create a temporary file to add
            temp_file = tmp_path / "temp_evil.txt"
            temp_file.write_text("I am evil")

            # Add it with a path traversal name: ../evil.txt relative to extract_dir
            # This attempts to write to tmp_path/evil.txt
            tar.add(temp_file, arcname="../evil.txt")

            temp_file.unlink()

        return tar_path, extract_dir, target_file

    def test_tar_slip_prevention(self, setup_malicious_tar):
        tar_path, extract_dir, target_file = setup_malicious_tar

        # Verify vulnerability is blocked
        # ModelManagement wraps TarError into ValueError
        with pytest.raises(ValueError, match="An error occurred while decompressing"):
            ModelManagement.decompress_to_cache(str(tar_path), str(extract_dir))

        # Double check that the file was NOT created outside
        assert not target_file.exists(), (
            "Path traversal vulnerability: file was extracted outside target directory!"
        )
