import pytest
from unittest.mock import patch
from pathlib import Path
from qwen3_embed.common.model_management import ModelManagement

def test_retrieve_model_gcs_missing_dir_raises_value_error(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Mock download_file_from_gcs to create a dummy tar.gz file
    def side_effect_download(url, output_path, show_progress=True):
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        return output_path

    # Mock decompress_to_cache to do NOTHING, simulating that the expected directory was not created.
    # This should trigger the failure because model_tmp_dir will not exist.

    with patch("qwen3_embed.common.model_management.ModelManagement.download_file_from_gcs", side_effect=side_effect_download),          patch("qwen3_embed.common.model_management.ModelManagement.decompress_to_cache"):

        with pytest.raises(ValueError, match="Could not find"):
            ModelManagement.retrieve_model_gcs(
                model_name="test-model",
                source_url="http://example.com/model.tar.gz",
                cache_dir=str(cache_dir)
            )
