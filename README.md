# Qwen3 Embed

**Lightweight Qwen3 text embedding and reranking via ONNX Runtime**

[![CI](https://github.com/n24q02m/qwen3-embed/actions/workflows/ci.yml/badge.svg)](https://github.com/n24q02m/qwen3-embed/actions/workflows/ci.yml)
[![Codecov](https://img.shields.io/codecov/c/github/n24q02m/qwen3-embed?logo=codecov&logoColor=white)](https://codecov.io/gh/n24q02m/qwen3-embed)
[![PyPI](https://img.shields.io/pypi/v/qwen3-embed?logo=pypi&logoColor=white)](https://pypi.org/project/qwen3-embed/)
[![License: Apache-2.0](https://img.shields.io/github/license/n24q02m/qwen3-embed)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](#)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005CED?logo=onnx&logoColor=white)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?logo=huggingface&logoColor=black)](#)
[![semantic-release](https://img.shields.io/badge/semantic--release-e10079?logo=semantic-release&logoColor=white)](https://github.com/python-semantic-release/python-semantic-release)
[![Renovate](https://img.shields.io/badge/renovate-enabled-1A1F6C?logo=renovatebot&logoColor=white)](https://developer.mend.io/)

Trimmed fork of [fastembed](https://github.com/qdrant/fastembed), keeping only Qwen3 models.

## Supported Models

### ONNX (default)

| Model | Type | Dims | Max Tokens | Size |
|-------|------|------|------------|------|
| `n24q02m/Qwen3-Embedding-0.6B-ONNX` | Embedding | 32-1024 (MRL) | 32768 | 573 MB |
| `n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16` | Embedding | 32-1024 (MRL) | 32768 | 517 MB |
| `n24q02m/Qwen3-Reranker-0.6B-ONNX` | Reranker | - | 40960 | 573 MB |
| `n24q02m/Qwen3-Reranker-0.6B-ONNX-Q4F16` | Reranker | - | 40960 | 518 MB |

### GGUF (optional, requires `llama-cpp-python`)

| Model | Type | Dims | Max Tokens | Size |
|-------|------|------|------------|------|
| `n24q02m/Qwen3-Embedding-0.6B-GGUF` | Embedding | 32-1024 (MRL) | 32768 | 378 MB |
| `n24q02m/Qwen3-Reranker-0.6B-GGUF` | Reranker | - | 40960 | 378 MB |

### HuggingFace Repos

| Format | Embedding | Reranker |
|--------|-----------|---------|
| ONNX | [n24q02m/Qwen3-Embedding-0.6B-ONNX](https://huggingface.co/n24q02m/Qwen3-Embedding-0.6B-ONNX) | [n24q02m/Qwen3-Reranker-0.6B-ONNX](https://huggingface.co/n24q02m/Qwen3-Reranker-0.6B-ONNX) |
| GGUF | [n24q02m/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/n24q02m/Qwen3-Embedding-0.6B-GGUF) | [n24q02m/Qwen3-Reranker-0.6B-GGUF](https://huggingface.co/n24q02m/Qwen3-Reranker-0.6B-GGUF) |

## Installation

```bash
pip install qwen3-embed

# For GGUF support
pip install qwen3-embed[gguf]
```

## Usage

### Text Embedding

```python
from qwen3_embed import TextEmbedding

# INT8 (default)
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX")

# Q4F16 (smaller, slightly less accurate)
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16")

# GGUF (requires: pip install qwen3-embed[gguf])
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF")

documents = [
    "Qwen3 is a multilingual embedding model.",
    "ONNX Runtime enables fast CPU inference.",
]

embeddings = list(model.embed(documents))
# Each embedding: numpy array of shape (1024,), L2-normalized

# Matryoshka Representation Learning (MRL) -- truncate to smaller dims
embeddings_256 = list(model.embed(documents, dim=256))
# Each embedding: numpy array of shape (256,), L2-normalized

# Query with instruction (for retrieval tasks)
queries = list(model.query_embed(
    ["What is Qwen3?"],
    task="Given a question, retrieve relevant passages",
))
```

### Reranking

```python
from qwen3_embed import TextCrossEncoder

reranker = TextCrossEncoder(model_name="n24q02m/Qwen3-Reranker-0.6B-ONNX")

query = "What is Qwen3?"
documents = [
    "Qwen3 is a series of large language models by Alibaba.",
    "The weather today is sunny.",
    "Qwen3-Embedding supports multilingual text embedding.",
]

scores = list(reranker.rerank(query, documents))
# scores: list of float in [0, 1], higher = more relevant

# Or rerank pairs directly
pairs = [
    ("What is AI?", "Artificial intelligence is a branch of computer science."),
    ("What is ML?", "Machine learning is a subset of AI."),
]
pair_scores = list(reranker.rerank_pairs(pairs))
```

## Key Features

- **Last-token pooling**: Uses the final token representation (with left-padding) instead of mean pooling.
- **MRL support**: Matryoshka Representation Learning allows truncating embeddings to any dimension from 32 to 1024 while preserving quality.
- **Instruction-aware**: Query embedding supports task instructions for better retrieval performance.
- **Causal LM reranking**: Reranker uses yes/no logit scoring via causal language model, producing calibrated [0, 1] scores.
- **Multiple backends**: ONNX Runtime (INT8, Q4F16) and GGUF (Q4_K_M via llama-cpp-python).
- **GPU optional, no PyTorch**: Runs on ONNX Runtime or llama-cpp-python -- no heavy ML framework required. Auto-detects GPU (CUDA, DirectML) when available.
- **Multilingual**: Both models support multi-language inputs.

## GPU Acceleration

Both ONNX and GGUF backends auto-detect GPU when available (`Device.AUTO` is the default).

### ONNX

Requires `onnxruntime-gpu` (CUDA) or `onnxruntime-directml` (Windows) instead of `onnxruntime`:

```bash
pip install onnxruntime-gpu  # NVIDIA CUDA
# or
pip install onnxruntime-directml  # Windows AMD/Intel/NVIDIA
```

```python
from qwen3_embed import TextEmbedding, Device

# Auto-detect GPU (default)
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX")

# Force CPU
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX", cuda=Device.CPU)

# Force CUDA
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX", cuda=Device.CUDA)
```

### GGUF

GPU is handled by `llama-cpp-python`. Install with CUDA support:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install qwen3-embed[gguf]
```

```python
from qwen3_embed import TextEmbedding, Device

# Auto-detect GPU (default, offloads all layers)
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF")

# Force CPU only
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF", cuda=Device.CPU)
```

## Development

```bash
mise run setup   # Install deps + pre-commit hooks
mise run lint    # ruff check + format --check
mise run test    # pytest
mise run fix     # ruff auto-fix + format
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache-2.0 - See [LICENSE](LICENSE). Original fastembed by [Qdrant](https://github.com/qdrant/fastembed).
