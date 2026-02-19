import os
import sys
import tempfile
from pathlib import Path

import pytest

from qwen3_embed.common.utils import define_cache_dir


def test_define_cache_dir_returns_secure_path_when_default():
    """Test that the default cache directory is not in the global temp directory."""
    # Unset the environment variable to test the default
    if "QWEN3_EMBED_CACHE_PATH" in os.environ:
        del os.environ["QWEN3_EMBED_CACHE_PATH"]

    cache_path = define_cache_dir()

    # It should NOT be in the global temp dir (e.g. /tmp)
    temp_dir = Path(tempfile.gettempdir())
    try:
        # relative_to raises ValueError if not a subpath
        cache_path.relative_to(temp_dir)
        is_in_temp = True
    except ValueError:
        is_in_temp = False

    assert not is_in_temp, f"Cache directory {cache_path} should not be in {temp_dir}"

    # It should be in the user's home directory (or equivalent)
    home = Path.home()

    if sys.platform == "linux":
        expected_base = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
        assert expected_base in cache_path.parents or expected_base == cache_path.parent
    elif sys.platform == "darwin":
        expected_base = home / "Library" / "Caches"
        assert expected_base in cache_path.parents or expected_base == cache_path.parent


def test_define_cache_dir_respects_env_var(monkeypatch):
    """Test that QWEN3_EMBED_CACHE_PATH overrides the default."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        monkeypatch.setenv("QWEN3_EMBED_CACHE_PATH", tmp_dir)
        cache_path = define_cache_dir()
        assert cache_path == Path(tmp_dir)


@pytest.mark.skipif(os.name != "posix", reason="Permission checks are POSIX-specific")
def test_define_cache_dir_sets_secure_permissions_on_posix():
    """Test that the cache directory has secure permissions (0o700)."""
    # Use a custom path to ensure we create it fresh
    with tempfile.TemporaryDirectory() as tmp_base:
        target_dir = os.path.join(tmp_base, "secure_cache")

        # Ensure it doesn't exist yet
        if os.path.exists(target_dir):
            os.rmdir(target_dir)

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("QWEN3_EMBED_CACHE_PATH", target_dir)
            path = define_cache_dir()

            assert path.exists()
            assert path.is_dir()

            # Check permissions
            st = os.stat(path)
            mode = st.st_mode & 0o777
            assert mode == 0o700, f"Permissions should be 0o700, got {oct(mode)}"
