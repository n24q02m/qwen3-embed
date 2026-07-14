import re

with open("qwen3_embed/rerank/cross_encoder/qwen3_cross_encoder.py", "r") as f:
    content = f.read()

# We want to iterate over FORBIDDEN_TOKENS rather than hardcoding.
# While hardcoding is a tiny bit faster, using `for token in FORBIDDEN_TOKENS:` is much safer.
