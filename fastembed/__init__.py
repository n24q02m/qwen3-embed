import importlib.metadata

from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed.text import TextEmbedding

try:
    version = importlib.metadata.version("qwen3-embed")
except importlib.metadata.PackageNotFoundError:
    version = "0.0.0"

__version__ = version
__all__ = [
    "TextEmbedding",
    "TextCrossEncoder",
]
