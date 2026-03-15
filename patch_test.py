import re

with open("tests/test_onnx_embedding.py", "r") as f:
    content = f.read()

new_content = re.sub(
    r'@patch\("qwen3_embed\.text\.onnx_embedding\.OnnxTextEmbedding"\)\ndef test_onnx_text_embedding_worker_init\(\n    mock_onnx_embedding: MagicMock,\n\) -> None:\n    OnnxTextEmbeddingWorker\(model_name=_MODEL_NAME, cache_dir="/tmp/cache", extra="arg"\)\n\n    mock_onnx_embedding\.assert_called_once_with\(\n        model_name=_MODEL_NAME, cache_dir="/tmp/cache", threads=1, extra="arg"\n    \)',
    '''@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding")
def test_onnx_text_embedding_worker_init(
    mock_onnx_embedding: MagicMock,
) -> None:
    worker = OnnxTextEmbeddingWorker.__new__(OnnxTextEmbeddingWorker)
    worker.init_embedding(model_name=_MODEL_NAME, cache_dir="/tmp/cache", extra="arg")

    mock_onnx_embedding.assert_called_once_with(
        model_name=_MODEL_NAME, cache_dir="/tmp/cache", threads=1, extra="arg"
    )''',
    content
)

with open("tests/test_onnx_embedding.py", "w") as f:
    f.write(new_content)
