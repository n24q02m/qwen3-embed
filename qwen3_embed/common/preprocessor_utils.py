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


def load_tokenizer(model_dir: Path) -> tuple[Tokenizer, dict[str, int]]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"Could not find config.json in {model_dir}")

    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise ValueError(f"Could not find tokenizer.json in {model_dir}")

    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise ValueError(f"Could not find tokenizer_config.json in {model_dir}")

    with open(str(config_path)) as config_file:
        config = json.load(config_file)

    with open(str(tokenizer_config_path)) as tokenizer_config_file:
        tokenizer_config = json.load(tokenizer_config_file)
        if "model_max_length" not in tokenizer_config and "max_length" not in tokenizer_config:
            raise ValueError("Models without model_max_length or max_length are not supported.")
        if "model_max_length" not in tokenizer_config:
            max_context = tokenizer_config["max_length"]
        elif "max_length" not in tokenizer_config:
            max_context = tokenizer_config["model_max_length"]
        else:
            max_context = min(tokenizer_config["model_max_length"], tokenizer_config["max_length"])

    tokens_map = load_special_tokens(model_dir)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=max_context)
    if not tokenizer.padding:
        pad_token_id = config.get("pad_token_id")
        if pad_token_id is None:
            pad_token_id = 0
        pad_token = tokenizer_config.get("pad_token", "")
        if isinstance(pad_token, dict):
            pad_token = pad_token.get("content", "")
        tokenizer.enable_padding(pad_id=pad_token_id, pad_token=pad_token)

    added_tokens: list[str | AddedToken] = []
    token_strings: list[str] = []
    for token in tokens_map.values():
        if isinstance(token, str):
            added_tokens.append(token)
            token_strings.append(token)
        elif isinstance(token, dict):
            added_tokens.append(AddedToken(**token))
            token_strings.append(token.get("content", ""))

    if added_tokens:
        tokenizer.add_special_tokens(added_tokens)

    special_token_to_id: dict[str, int] = {
        ts: tid for ts in token_strings if (tid := tokenizer.token_to_id(ts)) is not None
    }

    return tokenizer, special_token_to_id
