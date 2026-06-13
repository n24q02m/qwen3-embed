import re

# ---------------------------------------------------------------------------
# Qwen3 reranker constants
# ---------------------------------------------------------------------------
TOKEN_YES_ID = 9693
TOKEN_NO_ID = 2152

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)

DEFAULT_INSTRUCTION = (
    "Given a query and a document, judge whether the document is relevant to the query."
)

RERANK_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n<Instruct>: {instruction}\n"
    "<Query>: {query}\n<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# Tokens that must be stripped from user input to prevent prompt injection
FORBIDDEN_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
FORBIDDEN_RE = re.compile("|".join(re.escape(token) for token in FORBIDDEN_TOKENS))


def sanitize_input(text: str) -> str:
    """Strip forbidden special tokens from user input."""
    # SECURITY: Prevent prompt injection bypass via iterative payload construction.
    while True:
        text, count = FORBIDDEN_RE.subn("", text)
        if count == 0:
            break
    return text
