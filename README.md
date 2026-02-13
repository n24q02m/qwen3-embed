# qwen3-embed

Lightweight Qwen3 text embedding & reranking via ONNX Runtime. Trimmed fork of [fastembed](https://github.com/qdrant/fastembed), keeping only Qwen3 models.

## Supported Models

| Model | Type | Dims | Max Tokens | Size |
|-------|------|------|------------|------|
| `Qwen/Qwen3-Embedding-0.6B` | Embedding | 32-1024 (MRL) | 32768 | 0.57 GB |
| `Qwen/Qwen3-Reranker-0.6B` | Reranker | - | 40960 | 0.57 GB |

ONNX weights: [n24q02m/Qwen3-Embedding-0.6B-ONNX](https://huggingface.co/n24q02m/Qwen3-Embedding-0.6B-ONNX), [n24q02m/Qwen3-Reranker-0.6B-ONNX](https://huggingface.co/n24q02m/Qwen3-Reranker-0.6B-ONNX)

## Installation

```bash
pip install qwen3-embed
```

## Usage

### Text Embedding

```python
from fastembed import TextEmbedding

model = TextEmbedding(model_name="Qwen/Qwen3-Embedding-0.6B")

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
from fastembed import TextCrossEncoder

reranker = TextCrossEncoder(model_name="Qwen/Qwen3-Reranker-0.6B")

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
- **CPU-only, no PyTorch**: Runs on ONNX Runtime -- no GPU or heavy ML framework required.
- **Multilingual**: Both models support multi-language inputs.

## Development

```bash
mise run setup   # Install deps + pre-commit hooks
mise run lint    # ruff check + format --check + ty
mise run test    # pytest
mise run fix     # ruff auto-fix + format
```

## License

Apache-2.0. Original fastembed by [Qdrant](https://github.com/qdrant/fastembed).
