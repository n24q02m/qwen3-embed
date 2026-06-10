import io
import tarfile
from unittest.mock import MagicMock, patch

import pytest
import requests
from loguru import logger

from qwen3_embed.common.model_management import ModelManagement


class TestModelManagementExtra:
    """Extra tests to cover missing branches in model_management.py."""

    def test_get_session_init(self):
        """Test that _get_session initializes the session if it's None."""
        # Save original session to restore it later
        original_session = ModelManagement._session
        ModelManagement._session = None
        try:
            session = ModelManagement._get_session()
            assert isinstance(session, requests.Session)
            assert ModelManagement._session is session
            assert session.trust_env is False

            # Second call should return the same session
            session2 = ModelManagement._get_session()
            assert session2 is session
        finally:
            ModelManagement._session = original_session

    def test_decompress_absolute_path_mock(self, tmp_path):
        """Mock getmembers to return an absolute path to trigger line 371."""
        tar_path = tmp_path / "test.tar.gz"
        # Create a dummy tar file
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="safe.txt")
            info.size = 0
            tar.addfile(info, io.BytesIO(b""))

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Mock tarfile.open to return a tar with a malicious member
        mock_member = MagicMock()
        mock_member.name = "/absolute/path"

        with patch("tarfile.open") as mock_open:
            mock_tar = mock_open.return_value.__enter__.return_value
            mock_tar.__iter__.return_value = iter([mock_member])

            with pytest.raises(tarfile.TarError, match="Attempted path traversal"):
                ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

    def test_decompress_safe_symlink_and_hardlink(self, tmp_path):
        """Test safe symlinks and hardlinks to cover safe branch for links (line 399)."""
        cache_dir = tmp_path / "cache_links"
        cache_dir.mkdir()

        tar_path = tmp_path / "links.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Regular file
            info = tarfile.TarInfo(name="file.txt")
            info.size = 4
            tar.addfile(info, io.BytesIO(b"data"))

            # Safe symlink: points to a file within the same directory
            sym_info = tarfile.TarInfo(name="symlink.txt")
            sym_info.type = tarfile.SYMTYPE
            sym_info.linkname = "file.txt"
            tar.addfile(sym_info)

            # Safe hardlink: points to a file within the same directory (relative to root)
            hard_info = tarfile.TarInfo(name="hardlink.txt")
            hard_info.type = tarfile.LNKTYPE
            hard_info.linkname = "file.txt"
            tar.addfile(hard_info)

        result = ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))
        assert result == str(cache_dir)
        assert (cache_dir / "file.txt").exists()
        # On some systems/Python versions, symlink/hardlink might not be fully
        # supported or behaved differently in tests, but the logic should pass.
        assert (cache_dir / "symlink.txt").exists()
        assert (cache_dir / "hardlink.txt").exists()

    def test_decompress_no_data_filter(self, tmp_path):
        """Cover fallback by mocking tarfile to lack data_filter and verify manual extraction loop."""
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="file.txt")
            info.size = 0
            tar.addfile(info, io.BytesIO(b""))

        cache_dir = tmp_path / "cache_no_filter"
        cache_dir.mkdir()

        # We patch the tarfile module in the model_management namespace
        with patch("qwen3_embed.common.model_management.tarfile") as mock_tarfile_mod:
            # Setup mock_tar
            mock_tar = MagicMock()
            mock_tarfile_mod.open.return_value.__enter__.return_value = mock_tar

            # Mock getmembers to return a list of members
            member = MagicMock()
            member.name = "file.txt"
            member.isreg.return_value = True
            member.isdir.return_value = False
            member.issym.return_value = False
            member.islnk.return_value = False
            member.size = 0
            mock_tar.__iter__.return_value = iter([member])
            mock_tarfile_mod.TarError = tarfile.TarError

            # Ensure hasattr(tarfile, 'data_filter') returns False
            del mock_tarfile_mod.data_filter

            ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

            # Verify extract was called for the member (since extractall is no longer called in fallback)
            mock_tar.extract.assert_called_once_with(member, path=str(cache_dir))
            # Verify metadata sanitization
            # Since member.mode was a MagicMock, &= operation on it might be tricky to verify this way
            # In our code: member.mode &= 0o777 (which is member.mode = member.mode.__and__(0o777))
            # But member is also a MagicMock, so we can check attributes directly if we set them before
            assert member.uid == 0
            assert member.gid == 0
            assert member.uname == ""
            assert member.gname == ""

    def test_decompress_logging_on_error(self, tmp_path):
        """Verify logger.error is called on TarError."""
        tar_path = tmp_path / "corrupt.tar.gz"
        tar_path.write_text("not a tar file")

        cache_dir = tmp_path / "cache_err"
        cache_dir.mkdir()

        with patch.object(logger, "error") as mock_log_error:
            with pytest.raises(tarfile.TarError):
                ModelManagement.decompress_to_cache(str(tar_path), str(cache_dir))

            mock_log_error.assert_called()
            # Verify the log message contains the filename
            args, _ = mock_log_error.call_args
            assert str(tar_path) in args[0]

    @patch("qwen3_embed.common.model_management.model_info")
    def test_fetch_repo_files_no_sha_raises_value_error(self, mock_model_info):
        """Verify ValueError is raised if repo revision sha is None (line 289)."""
        mock_model_info.return_value = MagicMock(sha=None)
        repo = "org/repo"
        with pytest.raises(
            ValueError, match=f"Could not determine revision sha for repo '{repo}'"
        ):
            ModelManagement._fetch_repo_files(repo)
