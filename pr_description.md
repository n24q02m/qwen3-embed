🎯 What:
Refactored `_embed_documents` method in `qwen3_embed/text/onnx_text_model.py` to use `**kwargs` for optional parameters (e.g. `batch_size`, `parallel`, `providers`, `cuda`, `device_ids`, `local_files_only`, `specific_model_path`, `extra_session_options`) instead of explicit arguments.

💡 Why:
To address code health issue "Too many parameters", making the method signature cleaner and significantly improving the maintainability and readability of the `_embed_documents` API while avoiding breaking changes at call sites where args were passed as explicit keyword arguments and via `**kwargs`.

✅ Verification:
- Used `read_file` to confirm the method signature was applied properly and accurately.
- Used `uv run ruff check .` and `uv run ruff format .` to confirm that standard formatting and linting pass.
- Full test suite via `uv run pytest --ignore-glob='tests/test_integration*.py'` passed successfully without any regressions.
- `coverage run -m pytest` indicates 100% test coverage for the modified file.

✨ Result:
Cleaned up the `_embed_documents` method to have fewer explicit parameters using Python's standard keyword arguments grouping, resulting in better code readability without modifying behavior.
