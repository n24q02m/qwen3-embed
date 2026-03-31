## 2024-05-18 - [Fix Missing verify=True in requests.get]
**Vulnerability:** The code `requests.get` implicitly relies on the default configuration for TLS verification. This can be maliciously or accidentally disabled via the `REQUESTS_CA_BUNDLE` environment variable, leading to Man-In-The-Middle (MITM) attacks when downloading models from untrusted or tampered sources.
**Learning:** Explicit configuration (such as `verify=True` in HTTP requests) provides defense-in-depth against environment manipulation.
**Prevention:** Always explicitly set `verify=True` when executing HTTP requests to external or potentially unauthenticated sources using the `requests` library.
