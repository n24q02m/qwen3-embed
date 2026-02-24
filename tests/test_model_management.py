import unittest
from unittest.mock import MagicMock, patch, mock_open
import requests
import os
from qwen3_embed.common.model_management import ModelManagement

class TestModelManagement(unittest.TestCase):

    @patch("qwen3_embed.common.model_management.requests.get")
    @patch("qwen3_embed.common.model_management.tqdm")
    @patch("builtins.open", new_callable=mock_open)
    def test_download_file_from_gcs_success(self, mock_file, mock_tqdm, mock_get):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1024"}
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_get.return_value = mock_response

        # Call the method
        url = "http://example.com/file.tar.gz"
        output_path = "path/to/file.tar.gz"

        # Mock os.path.exists to return False so download proceeds
        with patch("os.path.exists", return_value=False):
            result = ModelManagement.download_file_from_gcs(url, output_path)

        # Assertions
        self.assertEqual(result, output_path)
        mock_get.assert_called_once_with(url, stream=True)
        mock_tqdm.assert_called_once()
        mock_file.assert_called_once_with(output_path, "wb")
        handle = mock_file()
        handle.write.assert_any_call(b"chunk1")
        handle.write.assert_any_call(b"chunk2")

    @patch("os.path.exists", return_value=True)
    def test_download_file_from_gcs_exists(self, mock_exists):
        url = "http://example.com/file.tar.gz"
        output_path = "path/to/file.tar.gz"

        result = ModelManagement.download_file_from_gcs(url, output_path)

        self.assertEqual(result, output_path)
        # Should not call requests.get if file exists
        # Since requests is not mocked here, if it was called it would likely fail or hang,
        # but to be safe we can mock it to ensure it's NOT called.
        with patch("qwen3_embed.common.model_management.requests.get") as mock_get:
             ModelManagement.download_file_from_gcs(url, output_path)
             mock_get.assert_not_called()

    @patch("qwen3_embed.common.model_management.requests.get")
    def test_download_file_from_gcs_403(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response

        url = "http://example.com/forbidden"
        output_path = "path/to/forbidden"

        with patch("os.path.exists", return_value=False):
            with self.assertRaises(PermissionError) as cm:
                ModelManagement.download_file_from_gcs(url, output_path)

            self.assertIn("Authentication Error", str(cm.exception))

    @patch("qwen3_embed.common.model_management.requests.get")
    @patch("qwen3_embed.common.model_management.tqdm")
    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print") # Capture print output
    def test_download_file_from_gcs_zero_content_length(self, mock_print, mock_file, mock_tqdm, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Missing content-length or 0
        mock_response.headers = {"content-length": "0"}
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        url = "http://example.com/zero"
        output_path = "path/to/zero"

        with patch("os.path.exists", return_value=False):
            ModelManagement.download_file_from_gcs(url, output_path)

        # Verify warning printed
        mock_print.assert_called_with(f"Warning: Content-length header is missing or zero in the response from {url}.")

        # tqdm should be disabled because total_size is 0
        # The implementation calculates show_progress = bool(total_size_in_bytes and show_progress)
        # So disable=True (since disable=not show_progress)
        args, kwargs = mock_tqdm.call_args
        self.assertTrue(kwargs.get('disable'))

    @patch("qwen3_embed.common.model_management.requests.get")
    @patch("qwen3_embed.common.model_management.tqdm")
    @patch("builtins.open", new_callable=mock_open)
    def test_download_file_from_gcs_no_progress(self, mock_file, mock_tqdm, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        url = "http://example.com/file"
        output_path = "path/to/file"

        with patch("os.path.exists", return_value=False):
            ModelManagement.download_file_from_gcs(url, output_path, show_progress=False)

        # tqdm should be disabled
        args, kwargs = mock_tqdm.call_args
        self.assertTrue(kwargs.get('disable'))
