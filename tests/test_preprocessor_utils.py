import json
import pytest
from pathlib import Path
from qwen3_embed.common.preprocessor_utils import load_special_tokens

class TestLoadSpecialTokens:
    """Tests for load_special_tokens utility."""

    def test_file_exists_and_valid(self, tmp_path: Path) -> None:
        """Test loading special tokens when the file exists and is valid JSON."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()
        tokens_map_path = model_dir / "special_tokens_map.json"

        expected_tokens = {"unk_token": "<unk>", "pad_token": "<pad>"}
        with open(tokens_map_path, "w") as f:
            json.dump(expected_tokens, f)

        result = load_special_tokens(model_dir)
        assert result == expected_tokens

    def test_file_missing(self, tmp_path: Path) -> None:
        """Test loading special tokens when the file is missing."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()

        result = load_special_tokens(model_dir)
        assert result == {}

    def test_file_invalid_json(self, tmp_path: Path) -> None:
        """Test loading special tokens when the file contains invalid JSON."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()
        tokens_map_path = model_dir / "special_tokens_map.json"

        with open(tokens_map_path, "w") as f:
            f.write("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            load_special_tokens(model_dir)
