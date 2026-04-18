## 2026-04-18 - SSRF via Open Redirects in requests.get
**Vulnerability:** Server-Side Request Forgery (SSRF) allowed by following arbitrary HTTP redirects in `requests.get` during GCS file downloads.
**Learning:** `requests.get` sets `allow_redirects=True` by default. Even if a URL is initially validated (e.g., `url.startswith("https://storage.googleapis.com/")`), the server can respond with a 3xx redirect to an internal or malicious endpoint, which `requests` will blindly follow.
**Prevention:** Explicitly pass `allow_redirects=False` to `requests.get()` when fetching resources from URLs constructed or influenced by external input. Manually check `response.status_code` to explicitly reject redirects (e.g., 301, 302) if they are not expected.
