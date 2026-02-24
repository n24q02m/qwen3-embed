## [2025-06-21] Prompt Injection in Reranker Template

**Vulnerability:** The Qwen3 reranker used a chat template directly formatted with user-supplied query and document strings. This allowed users to inject special tokens like `<|im_start|>` and `<|im_end|>` to break out of the template and potentially manipulate the model's output or context.

**Fix:** Introduced `FORBIDDEN_TOKENS` constant and `_sanitize_input` method in `Qwen3CrossEncoder` to strip `<|im_start|>`, `<|im_end|>`, and `<|endoftext|>` from `query`, `document`, and `instruction` before formatting.

**Lesson:** Always sanitize user input before inserting it into structured templates, especially for LLMs that rely on special tokens for control flow.
