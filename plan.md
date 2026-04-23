1. Modify `qwen3_embed/rerank/cross_encoder/qwen3_cross_encoder.py` to fix type checker errors.
   - Change `diff += 1.0` to `diff = diff + 1.0`
   - Cast `model_output` to `Any` or handle it so that `astype` doesn't fail.
2. Verify with `uv run ty check qwen3_embed` and `uv run pytest`.
3. Submit again.
