import os
import tempfile
from unittest import mock

from qwen3_embed.common.model_management import ModelManagement


class TestModelManagement:
    @mock.patch("qwen3_embed.common.model_management.requests.get")
    def test_download_file_from_gcs_timeout(self, mock_get):
        """Verify requests.get is called with a timeout."""
        # Setup mock response
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "10"}
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test_file")

            # Call the method
            ModelManagement.download_file_from_gcs(
                url="http://example.com/file", output_path=output_path, show_progress=False
            )

            # Assert requests.get was called with a timeout
            # This is the crucial check for the security fix
            args, kwargs = mock_get.call_args
            assert "timeout" in kwargs, "requests.get must be called with a 'timeout' parameter"
            assert kwargs["timeout"] is not None
            assert kwargs["timeout"] > 0
