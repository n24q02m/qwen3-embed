import json
from pathlib import Path

import pytest

from qwen3_embed.common.preprocessor_utils import load_special_tokens


class TestLoadSpecialTokens:
    def test_file_exists_and_valid(self, tmp_path: Path) -> None:
        """Test that load_special_tokens correctly loads a valid JSON file."""
        special_tokens = {"cls_token": "[CLS]", "sep_token": "[SEP]"}
        tokens_map_path = tmp_path / "special_tokens_map.json"

        with open(tokens_map_path, "w") as f:
            json.dump(special_tokens, f)

        result = load_special_tokens(tmp_path)
        assert result == special_tokens

    def test_file_missing(self, tmp_path: Path) -> None:
        """Test that load_special_tokens returns an empty dict when file is missing."""
        result = load_special_tokens(tmp_path)
        assert result == {}

    def test_file_invalid_json(self, tmp_path: Path) -> None:
        """Test that load_special_tokens raises JSONDecodeError for invalid JSON."""
        tokens_map_path = tmp_path / "special_tokens_map.json"

        with open(tokens_map_path, "w") as f:
            f.write("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            load_special_tokens(tmp_path)
