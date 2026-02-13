import importlib.metadata

from qwen3_embed.rerank.cross_encoder import TextCrossEncoder
from qwen3_embed.text import TextEmbedding

try:
    version = importlib.metadata.version("qwen3-embed")
except importlib.metadata.PackageNotFoundError:
    version = "0.0.0"

__version__ = version
__all__ = [
    "TextEmbedding",
    "TextCrossEncoder",
]
