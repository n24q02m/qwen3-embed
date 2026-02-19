import sys
import tempfile
from pathlib import Path

from qwen3_embed.common.utils import define_cache_dir


def test_cache_dir_creation_and_permissions(monkeypatch, tmp_path):
    """Test that define_cache_dir creates a directory with secure permissions."""

    # Use a custom cache path inside tmp_path
    cache_path = tmp_path / "custom_cache"
    monkeypatch.setenv("QWEN3_EMBED_CACHE_PATH", str(cache_path))

    returned_path = define_cache_dir()

    assert returned_path == cache_path
    assert returned_path.exists()
    assert returned_path.is_dir()

    if sys.platform != "win32":
        # Check permissions are 0o700 (rwx------)
        # Note: Depending on umask, it might be 700 or 750 or 755 by default.
        # But we want to ENFORCE 700 for security.
        mode = returned_path.stat().st_mode
        assert (mode & 0o777) == 0o700


def test_default_cache_location_is_secure(monkeypatch, tmp_path):
    """Test that default cache directory is not in /tmp and respects user directories."""

    monkeypatch.delenv("QWEN3_EMBED_CACHE_PATH", raising=False)

    # Mock user home directory
    mock_home = tmp_path / "home"
    mock_home.mkdir()
    monkeypatch.setenv("HOME", str(mock_home))
    monkeypatch.setenv("USERPROFILE", str(mock_home))  # Windows

    # Mock platform-specific env vars
    monkeypatch.setenv("XDG_CACHE_HOME", str(mock_home / ".cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(mock_home / "AppData/Local"))

    path = define_cache_dir()

    # Ensure it's inside our mocked home/cache location
    # resolve() handles symlinks to ensure robust check
    assert str(mock_home.resolve()) in str(path.resolve())

    # Ensure it's NOT the insecure default path
    insecure_default = Path(tempfile.gettempdir()) / "qwen3_embed_cache"
    assert path.resolve() != insecure_default.resolve()
