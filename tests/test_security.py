import tarfile
import pytest
import io
from qwen3_embed.common.model_management import ModelManagement

class TestSecurity:
    @pytest.fixture
    def malicious_tar(self, tmp_path):
        """Creates a tar file with a member trying to write to parent directory."""
        tar_path = tmp_path / "malicious.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add a file that tries to write to parent directory
            t = tarfile.TarInfo("../evil.txt")
            t.size = len(b"evil")
            tar.addfile(t, io.BytesIO(b"evil"))
        return tar_path

    def test_zip_slip_prevention(self, malicious_tar, tmp_path):
        """Tests that tar file extraction prevents Zip Slip attacks."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # We expect a ValueError because ModelManagement catches exceptions and raises ValueError
        # effectively wrapping the underlying tarfile error.
        # The underlying error is tarfile.OutsideDestinationError which contains "outside the destination"
        with pytest.raises(ValueError, match="outside the destination"):
             ModelManagement.decompress_to_cache(str(malicious_tar), str(cache_dir))

        # Verify evil.txt was NOT created in tmp_path (parent of cache)
        # Note: tmp_path is the base for 'cache', so ../evil.txt relative to cache
        # would be in tmp_path.
        assert not (tmp_path / "evil.txt").exists()
