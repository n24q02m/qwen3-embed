import json
from pathlib import Path
from typing import Any

from tokenizers import AddedToken, Tokenizer


def load_special_tokens(model_dir: Path) -> dict[str, Any]:
    tokens_map_path = model_dir / "special_tokens_map.json"
    if not tokens_map_path.exists():
        return {}

    with open(str(tokens_map_path)) as tokens_map_file:
        tokens_map = json.load(tokens_map_file)

    return tokens_map


def _read_json_config(model_dir: Path, filename: str) -> dict[str, Any]:
    file_path = model_dir / filename
    if not file_path.exists():
        raise ValueError(f"Could not find {filename} in {model_dir}")
    with open(str(file_path)) as f:
        return json.load(f)


def _get_max_context(tokenizer_config: dict[str, Any]) -> int:
    if "model_max_length" not in tokenizer_config and "max_length" not in tokenizer_config:
        raise ValueError("Models without model_max_length or max_length are not supported.")
    if "model_max_length" not in tokenizer_config:
        return tokenizer_config["max_length"]
    elif "max_length" not in tokenizer_config:
        return tokenizer_config["model_max_length"]
    else:
        return min(tokenizer_config["model_max_length"], tokenizer_config["max_length"])


def _configure_tokenizer_padding(
    tokenizer: Tokenizer, config: dict[str, Any], tokenizer_config: dict[str, Any]
) -> None:
    if not tokenizer.padding:
        pad_token_id = config.get("pad_token_id")
        if pad_token_id is None:
            pad_token_id = 0
        pad_token = tokenizer_config.get("pad_token", "")
        if isinstance(pad_token, dict):
            pad_token = pad_token.get("content", "")
        tokenizer.enable_padding(pad_id=pad_token_id, pad_token=pad_token)


def _add_special_tokens(tokenizer: Tokenizer, tokens_map: dict[str, Any]) -> dict[str, int]:
    for token in tokens_map.values():
        if isinstance(token, str):
            tokenizer.add_special_tokens([token])
        elif isinstance(token, dict):
            tokenizer.add_special_tokens([AddedToken(**token)])

    special_token_to_id: dict[str, int] = {}

    for token in tokens_map.values():
        if isinstance(token, str):
            special_token_to_id[token] = tokenizer.token_to_id(token)
        elif isinstance(token, dict):
            token_str = token.get("content", "")
            special_token_to_id[token_str] = tokenizer.token_to_id(token_str)

    return special_token_to_id


def load_tokenizer(model_dir: Path) -> tuple[Tokenizer, dict[str, int]]:
    config = _read_json_config(model_dir, "config.json")
    tokenizer_config = _read_json_config(model_dir, "tokenizer_config.json")

    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise ValueError(f"Could not find tokenizer.json in {model_dir}")

    max_context = _get_max_context(tokenizer_config)
    tokens_map = load_special_tokens(model_dir)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=max_context)

    _configure_tokenizer_padding(tokenizer, config, tokenizer_config)
    special_token_to_id = _add_special_tokens(tokenizer, tokens_map)

    return tokenizer, special_token_to_id
