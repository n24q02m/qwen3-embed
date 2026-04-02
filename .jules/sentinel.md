## 2024-05-18 - [Fix Missing verify=True in requests.get]
**Vulnerability:** The code `requests.get` implicitly relies on the default configuration for TLS verification. This can be maliciously or accidentally disabled via the `REQUESTS_CA_BUNDLE` environment variable, leading to Man-In-The-Middle (MITM) attacks when downloading models from untrusted or tampered sources.
**Learning:** Explicit configuration (such as `verify=True` in HTTP requests) provides defense-in-depth against environment manipulation.
**Prevention:** Always explicitly set `verify=True` when executing HTTP requests to external or potentially unauthenticated sources using the `requests library`.

## 2024-05-19 - [Test file metadata saving failure]
**Vulnerability:** Although not a direct security vulnerability, a failure to save metadata could lead to corrupted model verification in subsequent loads if errors are not handled.
**Learning:** Mocking low-level IO operations (like `pathlib.Path.write_text`) or serialization functions (like `json.dumps`) is an effective way to test error handling for edge cases such as full disks or invalid data.
**Prevention:** Ensure that critical IO operations are wrapped in `try-except` blocks and that failure to complete secondary tasks (like saving metadata) does not crash the main execution flow.
