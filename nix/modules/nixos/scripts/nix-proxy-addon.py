"""
mitmproxy addon for Nix fetch caching and logging.

- Caches responses by content hash
- Logs all fetches for attestation
- Enforces domain allowlist (optional)
"""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from mitmproxy import ctx, http

CACHE_DIR = Path(os.environ.get("NIX_PROXY_CACHE_DIR", "/var/cache/nix-proxy"))
LOG_DIR = Path(os.environ.get("NIX_PROXY_LOG_DIR", "/var/log/nix-proxy"))
ALLOWLIST = os.environ.get("NIX_PROXY_ALLOWLIST", "").split(",")
ALLOWLIST = [d.strip() for d in ALLOWLIST if d.strip()]


class NixProxyAddon:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = LOG_DIR / f"fetches-{datetime.now():%Y%m%d}.jsonl"

    def _hash_content(self, content: bytes) -> str:
        """SHA256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    def _cache_path(self, content_hash: str) -> Path:
        """Two-level cache path like git objects."""
        return CACHE_DIR / content_hash[:2] / content_hash[2:]

    def _log_fetch(self, url: str, content_hash: str, size: int, cached: bool):
        """Append fetch to log file."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "url": url,
            "sha256": content_hash,
            "size": size,
            "cached": cached,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _check_allowlist(self, host: str) -> bool:
        """Check if host is in allowlist (empty = allow all)."""
        if not ALLOWLIST:
            return True
        return any(
            host == allowed or host.endswith("." + allowed) for allowed in ALLOWLIST
        )

    def request(self, flow: http.HTTPFlow):
        """Check allowlist before forwarding request."""
        host = flow.request.host
        if not self._check_allowlist(host):
            flow.response = http.Response.make(
                403, f"Host {host} not in allowlist", {"Content-Type": "text/plain"}
            )
            ctx.log.warn(f"Blocked request to {host} (not in allowlist)")

    def response(self, flow: http.HTTPFlow):
        """Cache successful responses."""
        if flow.response.status_code != 200:
            return

        content = flow.response.content
        if not content:
            return

        content_hash = self._hash_content(content)
        cache_path = self._cache_path(content_hash)
        url = flow.request.pretty_url

        # Check if already cached
        cached = cache_path.exists()
        if not cached:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(content)
            ctx.log.info(f"Cached {url} -> {content_hash[:16]}... ({len(content)} bytes)")

        self._log_fetch(url, content_hash, len(content), cached)


addons = [NixProxyAddon()]
