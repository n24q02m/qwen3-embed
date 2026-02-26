import tarfile
import pytest
from qwen3_embed.common.model_management import ModelManagement

def test_tarfile_traversal_vulnerability(tmp_path):
    malicious_tar_path = tmp_path / "malicious.tar.gz"
    with tarfile.open(malicious_tar_path, "w:gz") as tar:
        t = tarfile.TarInfo(name="../pwned.txt")
        t.size = len(b"hacked")
        import io
        tar.addfile(t, io.BytesIO(b"hacked"))

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # The code currently catches TarError and re-raises as ValueError.
    # If filter='data' is missing (vulnerable), this might extract successfully
    # OR raise an error depending on the Python version's default behavior.
    # Python 3.14 (used here) likely defaults to 'data' filter (secure by default),
    # so the test passes because the vulnerability is mitigated by the runtime environment.

    # TO DEMONSTRATE THE VULNERABILITY (and the need for explicit protection on older Pythons),
    # we should check if we can bypass the check by forcing filter='fully_trusted' if possible,
    # or assert that we ARE explicitly setting filter='data' in the code.

    # Since we can't easily change the runtime version here to show failure,
    # we will modify the test to verify that filter='data' IS USED in the call.
    # But first, let's see if the current code fails WITHOUT the explicit filter on older Pythons.

    # Since we are on Python 3.14, 'data' is the default. The vulnerability exists for users on <3.14.
    # To properly test this, we should mock tarfile.open to ensure extractall is called with filter='data'.

    from unittest.mock import patch, MagicMock

    with patch("tarfile.open") as mock_open:
        mock_tar = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_tar

        ModelManagement.decompress_to_cache(str(malicious_tar_path), str(cache_dir))

        # This assertion should FAIL if we haven't implemented the fix yet.
        # The current code likely calls extractall(path=cache_dir) without filter='data'.
        mock_tar.extractall.assert_called_with(path=str(cache_dir), filter='data')
