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


def _get_max_context(tokenizer_config: dict[str, Any]) -> int:
    """Determine the maximum context length from tokenizer configuration."""
    if "model_max_length" not in tokenizer_config and "max_length" not in tokenizer_config:
        raise ValueError("Models without model_max_length or max_length are not supported.")

    if "model_max_length" not in tokenizer_config:
        return tokenizer_config["max_length"]

    if "max_length" not in tokenizer_config:
        return tokenizer_config["model_max_length"]

    return min(tokenizer_config["model_max_length"], tokenizer_config["max_length"])


def _configure_padding(
    tokenizer: Tokenizer, config: dict[str, Any], tokenizer_config: dict[str, Any]
) -> None:
    """Configure padding for the tokenizer if it is not already set."""
    if not tokenizer.padding:
        pad_token_id = config.get("pad_token_id", 0)
        pad_token = tokenizer_config.get("pad_token", "")
        if isinstance(pad_token, dict):
            pad_token = pad_token.get("content", "")
        tokenizer.enable_padding(pad_id=pad_token_id, pad_token=pad_token)


def _apply_special_tokens(tokenizer: Tokenizer, tokens_map: dict[str, Any]) -> dict[str, int]:
    """Apply special tokens to the tokenizer and return a mapping of tokens to IDs."""
    for token in tokens_map.values():
        if isinstance(token, str):
            tokenizer.add_special_tokens([token])
        elif isinstance(token, dict):
            tokenizer.add_special_tokens([AddedToken(**token)])

    special_token_to_id: dict[str, int] = {}
    for token in tokens_map.values():
        token_str = None
        if isinstance(token, str):
            token_str = token
        elif isinstance(token, dict):
            token_str = token.get("content", "")

        if token_str is not None:
            token_id = tokenizer.token_to_id(token_str)
            if token_id is not None:
                special_token_to_id[token_str] = token_id

    return special_token_to_id


def load_tokenizer(model_dir: Path) -> tuple[Tokenizer, dict[str, int]]:
    """Load tokenizer and special tokens from model directory."""
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"

    for path in [config_path, tokenizer_path, tokenizer_config_path]:
        if not path.exists():
            raise ValueError(f"Could not find {path.name} in {model_dir}")

    with open(str(config_path)) as config_file:
        config = json.load(config_file)

    with open(str(tokenizer_config_path)) as tokenizer_config_file:
        tokenizer_config = json.load(tokenizer_config_file)

    max_context = _get_max_context(tokenizer_config)
    tokens_map = load_special_tokens(model_dir)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=max_context)

    _configure_padding(tokenizer, config, tokenizer_config)
    special_token_to_id = _apply_special_tokens(tokenizer, tokens_map)

    return tokenizer, special_token_to_id
