# Witness Proxy Demo

This demonstrates the Armitage witness proxy - a pure Haskell replacement for
the Python mitmproxy-based implementation.

## What The Proxy Does

1. Intercepts HTTP/HTTPS traffic (TLS MITM with generated certificates)
2. Caches responses by content hash (SHA256, two-level directories)
3. Logs all fetches as attestations (JSONL format)
4. Optional domain allowlist for policy enforcement

## The Old Way (Python mitmproxy)

The NixOS module at `nix/modules/nixos/nix-proxy.nix` used mitmproxy with a
Python addon script (`nix/modules/nixos/scripts/nix-proxy-addon.py`).

### Problems with the old approach

1. **Python dependency** - mitmproxy pulls in ~100+ Python packages
2. **Startup time** - Python interpreter initialization adds latency
3. **No type safety** - Runtime errors from typos, missing attrs
4. **Complex deployment** - Need Python environment in containers
5. **No gRPC** - Can't integrate with NativeLink CAS
6. **External dependency** - mitmproxy is a large, moving target

### Old proxy usage (for reference)

```bash
# Requires mitmproxy in environment
mitmdump \
  --listen-host 127.0.0.1 \
  --listen-port 8080 \
  --set confdir=/tmp/nix-proxy/certs \
  --scripts nix/modules/nixos/scripts/nix-proxy-addon.py
```

---

## The New Way (Armitage Haskell)

The new implementation is pure Haskell using the crypton ecosystem.

### Build

```bash
# Build with Buck2
buck2 build //armitage/proxy:armitage-proxy

# Or with nix
nix develop -c buck2 build //armitage/proxy:armitage-proxy
```

### Start the proxy

```bash
# Create directories
mkdir -p /tmp/armitage/{cache,logs,certs}

# Set environment and run
PROXY_PORT=8080 \
PROXY_CACHE_DIR=/tmp/armitage/cache \
PROXY_LOG_DIR=/tmp/armitage/logs \
PROXY_CERT_DIR=/tmp/armitage/certs \
  buck2 run //armitage/proxy:armitage-proxy
```

Output:
```
======================================
  Armitage Witness Proxy (TLS MITM)
======================================
Port:      8080
Cache:     /tmp/armitage/cache
Logs:      /tmp/armitage/logs
Certs:     /tmp/armitage/certs
Allowlist: 0 domains
Generating CA certificate...
CA certificate written to: /tmp/armitage/certs/ca.pem

Trust the CA certificate at: /tmp/armitage/certs/ca.pem

Listening on :8080
```

### Test with curl

```bash
# HTTP request
curl -x http://127.0.0.1:8080 http://example.com

# HTTPS request (trusting Armitage CA)
curl -x http://127.0.0.1:8080 \
     --cacert /tmp/armitage/certs/ca.pem \
     https://example.com
```

### Check the cache

```bash
# List cached files
find /tmp/armitage/cache -type f | head

# View attestation log
cat /tmp/armitage/logs/fetches.jsonl | jq .
```

### Proxy Output (HTTP)

```
GET example.com
CACHED: 2c9530ee6c979e2c812a6227e199b50af9a21af5390cf731083dff2328c0b61a
```

### Proxy Output (HTTPS)

```
CONNECT example.com:443
HTTPS GET https://example.com/
HTTPS CACHED: 2c9530ee6c979e2c812a6227e199b50af9a21af5390cf731083dff2328c0b61a
```

Note: Same SHA256 hash for both - content-addressing works regardless of protocol.

### Attestation Log

```json
{
  "url": "http://example.com:80/",
  "host": "example.com",
  "sha256": "2c9530ee6c979e2c812a6227e199b50af9a21af5390cf731083dff2328c0b61a",
  "size": 828,
  "timestamp": "2026-01-25T19:43:22.832483Z",
  "method": "GET",
  "cached": false
}
{
  "url": "https://example.com/",
  "host": "example.com", 
  "sha256": "2c9530ee6c979e2c812a6227e199b50af9a21af5390cf731083dff2328c0b61a",
  "size": 828,
  "timestamp": "2026-01-25T19:43:23.292492Z",
  "method": "GET",
  "cached": false
}
```

### Cache Directory

```
/tmp/armitage/cache/2c/9530ee6c979e2c812a6227e199b50af9a21af5390cf731083dff2328c0b61a
```

Two-level directory structure like git objects (`2c/9530ee...`).

### Advantages of the new proxy

1. **Single static binary** - No runtime dependencies
2. **Fast startup** - Compiled, no interpreter
3. **Type safe** - Compiler catches errors
4. **Simple deployment** - One binary in container
5. **gRPC ready** - grapesy for NativeLink CAS integration
6. **Crypton ecosystem** - Modern, maintained TLS/X.509 libraries

---

## Demo 3: Side-by-Side Comparison

### Fetch the same URL through both proxies

```bash
# Terminal 1: Old proxy on port 8080
NIX_PROXY_CACHE_DIR=/tmp/old/cache \
NIX_PROXY_LOG_DIR=/tmp/old/logs \
mitmdump --listen-port 8080 --set confdir=/tmp/old/certs \
  --scripts nix/modules/nixos/scripts/nix-proxy-addon.py

# Terminal 2: New proxy on port 8081
PROXY_PORT=8081 \
PROXY_CACHE_DIR=/tmp/new/cache \
PROXY_LOG_DIR=/tmp/new/logs \
PROXY_CERT_DIR=/tmp/new/certs \
  buck2 run //armitage/proxy:armitage-proxy

# Terminal 3: Fetch through both
URL="https://raw.githubusercontent.com/NixOS/nixpkgs/master/README.md"

curl -s -x http://127.0.0.1:8080 --cacert /tmp/old/certs/mitmproxy-ca-cert.pem "$URL" | sha256sum
curl -s -x http://127.0.0.1:8081 --cacert /tmp/new/certs/ca.pem "$URL" | sha256sum
```

Both should produce identical SHA256 hashes - the content is the same.

### Compare attestation formats

```bash
# Old format
cat /tmp/old/logs/fetches-*.jsonl | jq .

# New format
cat /tmp/new/logs/fetches.jsonl | jq .
```

The new format includes additional fields (`method`, `host`) for richer attestations.

---

## Demo 4: Integration with Nix Build

### Configure nix to use the proxy

```bash
# Start Armitage proxy
PROXY_PORT=8080 \
PROXY_CACHE_DIR=/tmp/armitage/cache \
PROXY_LOG_DIR=/tmp/armitage/logs \
PROXY_CERT_DIR=/tmp/armitage/certs \
  buck2 run //armitage/proxy:armitage-proxy &

# Run a nix build with proxy
http_proxy=http://127.0.0.1:8080 \
https_proxy=http://127.0.0.1:8080 \
SSL_CERT_FILE=/tmp/armitage/certs/ca.pem \
NIX_SSL_CERT_FILE=/tmp/armitage/certs/ca.pem \
  nix build nixpkgs#hello --rebuild

# Check what was fetched
cat /tmp/armitage/logs/fetches.jsonl | jq -r '.url' | sort -u
```

### Expected output

```
https://cache.nixos.org/nar/...
https://cache.nixos.org/...narinfo
```

All fetches are now witnessed and content-addressed cached.

---

## Demo 5: Domain Allowlist

### Restrict to specific domains

```bash
PROXY_PORT=8080 \
PROXY_CACHE_DIR=/tmp/armitage/cache \
PROXY_LOG_DIR=/tmp/armitage/logs \
PROXY_CERT_DIR=/tmp/armitage/certs \
PROXY_ALLOWLIST="cache.nixos.org,github.com,githubusercontent.com" \
  buck2 run //armitage/proxy:armitage-proxy
```

### Test allowed and blocked

```bash
# Allowed
curl -x http://127.0.0.1:8080 \
     --cacert /tmp/armitage/certs/ca.pem \
     https://cache.nixos.org/nix-cache-info

# Blocked
curl -x http://127.0.0.1:8080 \
     --cacert /tmp/armitage/certs/ca.pem \
     https://evil.com/malware.sh
```

The second request returns `403 Forbidden`.

---

## Next Steps

### Phase 1.5: NativeLink CAS Integration

Instead of writing to local filesystem:

```haskell
-- Current: local cache
cacheStore :: FilePath -> ContentHash -> ByteString -> IO ()

-- Future: NativeLink CAS
cacheToNativeLink :: CASClient -> Digest -> ByteString -> IO ()
```

With grapesy now building on GHC 9.12, this is ready to implement.

### Phase 1.6: Nix Binary Cache Facade

Expose NativeLink CAS as a nix substituter:

```
GET /<hash>.narinfo  →  CAS lookup: narinfo/<hash>
GET /nar/<hash>.nar  →  CAS lookup: nar/<hash>
```

This allows `nix build` to substitute directly from our CAS.
