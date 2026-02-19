from pathlib import Path

import pytest

from qwen3_embed.common.preprocessor_utils import load_tokenizer


def test_load_tokenizer_missing_config(tmp_path: Path):
    with pytest.raises(ValueError, match="Could not find config.json"):
        load_tokenizer(tmp_path)

def test_load_tokenizer_missing_tokenizer_json(tmp_path: Path):
    (tmp_path / "config.json").touch()
    with pytest.raises(ValueError, match="Could not find tokenizer.json"):
        load_tokenizer(tmp_path)

def test_load_tokenizer_missing_tokenizer_config(tmp_path: Path):
    (tmp_path / "config.json").touch()
    (tmp_path / "tokenizer.json").touch()
    with pytest.raises(ValueError, match="Could not find tokenizer_config.json"):
        load_tokenizer(tmp_path)
