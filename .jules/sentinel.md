## 2024-05-18 - [Fix Missing verify=True in requests.get]
**Vulnerability:** The code `requests.get` implicitly relies on the default configuration for TLS verification. This can be maliciously or accidentally disabled via the `REQUESTS_CA_BUNDLE` environment variable, leading to Man-In-The-Middle (MITM) attacks when downloading models from untrusted or tampered sources.
**Learning:** Explicit configuration (such as `verify=True` in HTTP requests) provides defense-in-depth against environment manipulation.
**Prevention:** Always explicitly set `verify=True` when executing HTTP requests to external or potentially unauthenticated sources using the `requests` library.

## 2024-05-18 - [Enforce HTTPS for Model Downloads]
**Vulnerability:** The code allowed downloading models over unencrypted HTTP connections from GCS (`http://storage.googleapis.com/`), which is vulnerable to Man-In-The-Middle (MITM) attacks where an attacker could intercept and tamper with model weights or metadata in transit.
**Learning:** Always strictly validate URL schemes and enforce HTTPS for fetching resources over networks, especially sensitive artifacts like model weights that can execute malicious code if tampered with.
**Prevention:** Restrict allowed download schemes to `https://` only and reject `http://` URLs explicitly.
