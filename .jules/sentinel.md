## 2024-05-28 - [Fix Prompt Injection Bypass in Qwen3CrossEncoder]
**Vulnerability:** The `_sanitize_input` method in `Qwen3CrossEncoder` only performed a single pass to strip forbidden tokens (like `<|im_start|>`). This allowed prompt injection bypass via iterative payload construction, where an attacker could provide nested tokens (e.g., `<|<|im_start|>im_start|>`) that resolve to forbidden tokens after the single pass.
**Learning:** Security validation and sanitization loops must continue until no further instances of forbidden tokens are found, as attackers can construct payloads that resolve dynamically during processing.
**Prevention:** Always use a `while` loop combined with `any()` to iteratively strip forbidden tokens until the payload is entirely clean.

## 2024-05-18 - [Fix Missing verify=True in requests.get]
**Vulnerability:** The code `requests.get` implicitly relies on the default configuration for TLS verification. This can be maliciously or accidentally disabled via the `REQUESTS_CA_BUNDLE` environment variable, leading to Man-In-The-Middle (MITM) attacks when downloading models from untrusted or tampered sources.
**Learning:** Explicit configuration (such as `verify=True` in HTTP requests) provides defense-in-depth against environment manipulation.
**Prevention:** Always explicitly set `verify=True` when executing HTTP requests to external or potentially unauthenticated sources using the `requests` library.