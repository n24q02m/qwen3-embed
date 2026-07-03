## YYYY-MM-DD - SSRF Prevention with Hostname vs Netloc
**Vulnerability:** `urllib.parse.urlparse(url).netloc` was used instead of `.hostname` for URL domain validation.
**Learning:** Checking `.netloc` instead of `.hostname` can sometimes lead to SSRF bypasses if the check uses `.endswith()` or similar methods, because `.netloc` includes userinfo and port syntax (e.g., `user:pass@host:port`). Although the original exact match was strictly safe, using `.hostname` is the robust standard that prevents future regression to bypassable patterns and handles explicit ports cleanly.
**Prevention:** Always use `urllib.parse.urlparse(url).hostname` for URL validation when verifying the domain.
