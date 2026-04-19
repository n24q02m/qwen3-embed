## 2024-05-18 - [Fix Missing verify=True in requests.get]
**Vulnerability:** The code `requests.get` implicitly relies on the default configuration for TLS verification. This can be maliciously or accidentally disabled via the `REQUESTS_CA_BUNDLE` environment variable, leading to Man-In-The-Middle (MITM) attacks when downloading models from untrusted or tampered sources.
**Learning:** Explicit configuration (such as `verify=True` in HTTP requests) provides defense-in-depth against environment manipulation.
**Prevention:** Always explicitly set `verify=True` when executing HTTP requests to external or potentially unauthenticated sources using the `requests` library.

## 2024-05-20 - [Fix SSRF via Open Redirects in GCS Downloads]
**Vulnerability:** Server-Side Request Forgery (SSRF) was possible because HTTP requests followed redirects by default. A malicious or compromised server could redirect to internal resources (e.g., metadata endpoints, internal network IPs) circumventing initial URL validation.
**Learning:** Initial URL validation is insufficient if the HTTP client automatically follows redirects to unvalidated destinations.
**Prevention:** Always set `allow_redirects=False` in HTTP clients and explicitly validate or reject redirect status codes (301, 302, 303, 307, 308) to prevent SSRF via open redirects.
