"""Tests for secure cache directory creation."""

import os
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

from qwen3_embed.common.utils import define_cache_dir


class TestSecureCache:
    """Tests for secure cache directory creation."""

    @pytest.fixture
    def mock_env(self):
        """Mock environment variables."""
        with mock.patch.dict(os.environ, {}, clear=True):
            yield

    def test_default_cache_dir_is_not_tmp(self, mock_env):
        """Ensure default cache dir is not in /tmp."""
        with (
            mock.patch("tempfile.gettempdir", return_value="/tmp/mocked_temp"),
            mock.patch("pathlib.Path.home", return_value=Path("/home/mockuser")),
            mock.patch("sys.platform", "linux"),
            mock.patch("pathlib.Path.mkdir"),
            mock.patch("pathlib.Path.chmod"),
        ):
            path = define_cache_dir()
            assert "/tmp/mocked_temp" not in str(path)
            assert str(path) == "/home/mockuser/.cache/qwen3_embed"

    def test_respects_env_var(self):
        """Ensure QWEN3_EMBED_CACHE_PATH is respected."""
        custom_path = "/tmp/custom_cache_location"
        with (
            mock.patch.dict(os.environ, {"QWEN3_EMBED_CACHE_PATH": custom_path}),
            mock.patch("pathlib.Path.mkdir"),
        ):
            path = define_cache_dir()
            assert str(path) == custom_path

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Permission checks are different on Windows"
    )
    def test_directory_permissions(self, tmp_path):
        """Ensure directory has secure permissions (0o700) on Unix."""
        test_cache = tmp_path / "secure_cache_test"
        if test_cache.exists():
            shutil.rmtree(test_cache)

        path = define_cache_dir(str(test_cache))
        assert path.exists()
        stat = path.stat()
        assert (stat.st_mode & 0o777) == 0o700

    def test_platform_specific_defaults_linux(self, mock_env):
        """Test Linux default."""
        with (
            mock.patch("sys.platform", "linux"),
            mock.patch("pathlib.Path.home", return_value=Path("/home/user")),
            mock.patch("pathlib.Path.mkdir"),
            mock.patch("pathlib.Path.chmod"),
        ):
            path = define_cache_dir()
            assert str(path) == "/home/user/.cache/qwen3_embed"

    def test_platform_specific_defaults_macos(self, mock_env):
        """Test macOS default."""
        with (
            mock.patch("sys.platform", "darwin"),
            mock.patch("pathlib.Path.home", return_value=Path("/Users/user")),
            mock.patch("pathlib.Path.mkdir"),
            mock.patch("pathlib.Path.chmod"),
        ):
            path = define_cache_dir()
            assert str(path) == "/Users/user/Library/Caches/qwen3_embed"

    def test_platform_specific_defaults_windows(self, mock_env):
        """Test Windows default."""
        expected_path = "C:/Users/User/AppData/Local/qwen3_embed"

        with (
            mock.patch("sys.platform", "win32"),
            mock.patch.dict(os.environ, {"LOCALAPPDATA": "C:/Users/User/AppData/Local"}),
            mock.patch("pathlib.Path.mkdir"),
        ):
            path = define_cache_dir()
            assert str(path).replace("\\", "/") == expected_path
