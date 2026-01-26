# Armitage: The Nix Compatibility Layer

## Overview

Armitage is the shim between typed builds and the nix daemon. Named for the man
who put the team together - the one who orchestrated the run before anyone knew
what the real job was.

Armitage:

- Forces CA-derivations through the daemon
- Witnesses all fetches into R2
- Records attestations with *coeffect discharge proofs*
- Bridges typed Dhall builds to nix infrastructure
- Ties out to professional-grade backends (NativeLink) and
  extreme redunancy stores (R2)
- Runs in user space
- Proves all store operats are logically monotone
- Shrinks back down to a network witness

It has a kick at first. Then you realize what it's for.

## The Category Error

The travesty of `nixpkgs`: confusing build environment with runtime environment.

A program built in Ubuntu doesn't need to run in Ubuntu. An adult doesn't live
(by any necessilty) where they were born. The build setting and run setting 
are orthogonal.

```
WRONG (nixpkgs):
  Build in stdenv → Run in same stdenv
  Everything rebuilds when stdenv changes
  glibc update = world rebuild

RIGHT (armitage):
  Build in OCI container (Ubuntu-like, "NixOS FHS 26.04")
  Extract artifacts
  Run anywhere (autopatchelf, minimal runtime)
```

The daemon's model of "purity" is wrong. We route around it.

## Coeffects: What Builds Actually Need

"Purity" is a boolean. That's too coarse. The real question: what does this
build *require* from the environment?

### The Resource Algebra

```dhall
let Resource = 
  < Pure                           -- needs nothing external
  | Network                        -- needs network access
  | Auth : Text                    -- needs credential (e.g., "huggingface")
  | Sandbox : Text                 -- needs isolation (e.g., "gpu")
  | Filesystem : Text              -- needs filesystem path
  | Both : Resource → Resource → Resource
  >

-- Combine resources
let _⊗_ : Resource → Resource → Resource
    = λ(r : Resource) → λ(s : Resource) → Resource.Both r s

-- Examples
let needs-network = Resource.Network
let needs-hf-auth = Resource.Network ⊗ Resource.Auth "huggingface"
let needs-gpu-sandbox = Resource.Sandbox "gpu"
```

### Builds Declare Requirements

```dhall
let Build = ./Build.dhall
let Resource = ./Resource.dhall
let Toolchain = ./Toolchain.dhall
let Triple = ./Triple.dhall
let CFlags = ./CFlags.dhall

in Build.target {
  name = "llama-inference",
  srcs = ["src/main.cpp", "src/model.cpp"],
  deps = [":llama-cpp"],
  
  toolchain = Toolchain.clang {
    version = "18",
    host = Triple.x86_64-linux-gnu,
    target = Triple.x86_64-linux-gnu,
    cflags = [CFlags.opt.O3, CFlags.std.cxx20]
  },
  
  -- The theory is visible
  requires = Resource.network 
           ⊗ Resource.auth "huggingface"
           ⊗ Resource.sandbox "gpu"
}
```

The `requires` field is the coeffect. It says exactly what external resources
this build needs. Not a boolean "pure/impure" - a typed, composable algebra.

## The Witness Proxy

Armitage intercepts all network requests. Not to block - to **witness**.

```
Build → Armitage Proxy → Network
              │
              └→ Log: { url, identity, timestamp, response_hash }
                      │
                      └→ CAS into R2
```

Every fetch becomes legible:
- What was fetched
- By whom (ed25519 identity)
- When
- What came back (content hash)

The proxy doesn't enforce policy. It produces **evidence**.

### First Fetch, Forever Cached

```
First fetch:
  Build → Proxy → github.com/foo/bar@v1.0.0
                      │
                      ├→ content: sha256:abc...
                      ├→ R2: PUT sha256:abc... (the bytes)
                      └→ Attestation: {
                           url: "github.com/foo/bar@v1.0.0",
                           content: sha256:abc,
                           fetched_by: ed25519:xyz,
                           timestamp: ...
                         }

Future fetch (anyone):
  Build → Proxy → sha256:abc...
                      │
                      └→ R2: GET sha256:abc...

  GitHub is not consulted. The hash is the artifact.
```

## Attestation as Coeffect Discharge

When you *run* a build with `requires = Network ⊗ Auth "hf"`, you must
**discharge** the coeffect. The discharge becomes the attestation:

```
Attestation = {
  content      : Hash,              -- what was produced
  coeffects    : Resource,          -- what was required  
  discharged   : DischargeProof,    -- how it was satisfied
  identity     : Ed25519PublicKey,  -- who ran it
  signature    : Signature          -- proof of attestation
}

DischargeProof = {
  network      : [NetworkAccess],   -- URLs fetched, response hashes
  auth         : [AuthProof],       -- credential hashes (not secrets)
  sandbox      : SandboxConfig      -- isolation used
}
```

The attestation records exactly how the coeffects were discharged. Future
verifiers can check:

1. Was the content hash correct?
2. Were the coeffects discharged legitimately?
3. Do I trust this identity's attestations?

## Trust Policy

Trust is explicit, not hidden in sandbox flags:

```dhall
let TrustPolicy = 
  < TrustSelf                       -- I built it
  | TrustIdentity : Ed25519Key      -- Trust this key
  | TrustOrg : Text                 -- Trust org's keys
  | TrustMultiple : Natural         -- Require N independent attestations
  | TrustReproducible               -- Multiple attestations, same hash
  >

let myPolicy = TrustPolicy.TrustMultiple 2  -- require 2 witnesses
```

Cache lookup requires content match AND trust policy:

```
lookup : Hash → TrustPolicy → Maybe Artifact
lookup hash policy =
  attestations ← getAttestations hash
  if any (satisfies policy) attestations
    then Just (fetchFromR2 hash)
    else Nothing
```

## The Daemon Problem

The nix daemon exists to "do things as root." It was proposed at a time when
user namespaces (`unshare`) were poorly understood - roughly as poorly understood
as linking is today, if `/run/opengl-driver` is anything to go by.

The same is true of the Docker daemon. Both were hacks for privilege escalation
that got captured almost immediately. The "trusted users" model, the socket
access games, the complexity that creates dependency on insiders - these aren't
bugs. They're features for incumbents.

The daemon isn't vestigial. It's **hostile infrastructure**.

### What The Daemon Actually Does

| Claimed Function | Reality |
|------------------|---------|
| "Multi-user safety" | Creates attack surface via setuid, socket ACLs |
| "Store coordination" | Unnecessary - CAS is naturally idempotent |
| "Sandbox enforcement" | Poorly implemented, bypassable, theater |
| "Cache coordination" | Could be stateless HTTP, isn't |
| "GC coordination" | Creates complexity to justify daemon existence |

### What We Actually Need

| Need | Solution |
|------|----------|
| Privilege for `/nix/store` writes | User namespaces (`unshare -r`) |
| Isolation | Bubblewrap, user namespaces |
| Network witnessing | Armitage proxy (content-addressed) |
| Cache | Stateless CAS (NativeLink + R2) |
| GC | Don't. Storage is cheap. Or refcount. |

### The Division of Labor

For now, we route around the daemon where possible:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Armitage Territory                               │
├─────────────────────────────────────────────────────────────────────────┤
│  • All network fetches (witnessed, content-addressed)                    │
│  • Build artifact storage (NativeLink CAS)                               │
│  • Attestations and trust policy                                         │
│  • Remote execution (Buck2 RE)                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ (minimal surface)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Daemon Territory (Shrinking)                        │
├─────────────────────────────────────────────────────────────────────────┤
│  • /nix/store writes (until we have our own store)                       │
│  • NAR unpacking (until we have hnix-store)                              │
│  • Signature verification (until we have our own trust)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

The goal is to shrink daemon territory to zero. Not because we're ideological
about it, but because every daemon interaction is an opportunity for:
- Policy enforcement we didn't ask for
- Complexity that serves maintainers, not users
- Attack surface that benefits adversaries

### Practical Migration

**Phase 1 (Now):** Armitage witnesses fetches, daemon does builds
- Proxy intercepts all network during FOD builds
- Content goes to CAS, daemon gets file:// URLs
- Daemon still writes to `/nix/store`

**Phase 2:** Armitage provides binary cache
- NativeLink CAS serves as nix substituter
- Daemon fetches from our infrastructure
- We control what gets substituted

**Phase 3:** Armitage replaces store writes
- User namespace store (`/home/$USER/.nix`)
- No daemon for single-user case
- Daemon only for legacy multi-user compat

**Phase 4:** Daemon-free operation
- Full build pipeline without daemon
- OCI containers with bundled store
- The daemon becomes optional, then forgotten

## Agent Memory

R2 isn't just a cache. It's **agent memory**.

```
Session 1:
  Agent figures something out
  Agent stores: R2:sha256:abc
  Attestation: "I (ed25519:agent-xyz) produced this"
  Session ends

Session 2 (different session, same or different agent):
  Agent recalls: sha256:abc
  It's there. The work compounds.
```

The content-addressed store is exocortex. The attestation is autobiography.
Agents with keys can persist knowledge beyond the context window.

This is the cure for induced Korsakov's.

## Implementation

### Phase 1: Witness Proxy ✓

**Status: Implemented** - `armitage/proxy/Main.hs`

Pure Haskell MITM proxy using the crypton ecosystem. Replaces the prototype
Python mitmproxy with a proper implementation.

#### Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│                 │     │                      │     │                 │
│  Nix Build      │────▶│  Armitage Witness    │────▶│  Origin Server  │
│  (curl, wget)   │     │  Proxy               │     │                 │
│                 │◀────│                      │◀────│                 │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                               │
                               │ writes
                               ▼
                        ┌──────────────────┐
                        │ Content Cache    │  (SHA256-addressed)
                        │ Attestation Log  │  (JSONL)
                        └──────────────────┘
```

#### Certificate Authority

On first run, generates a self-signed CA:
- RSA 2048-bit key, 10-year validity
- Stored at `$PROXY_CERT_DIR/ca.pem` and `ca-key.pem`
- Builds trust via `SSL_CERT_FILE`

#### TLS MITM Flow

1. Client sends `CONNECT example.com:443`
2. Proxy responds `200 Connection Established`
3. Proxy generates certificate for host, signed by CA
4. Dual TLS handshake: server-to-client, client-to-origin
5. HTTP flows over intercepted tunnel, cached and logged

#### Certificate Signing

Uses `objectToSignedExact` from x509 with PKCS#15 RSA:

```haskell
signCertificate :: PrivateKey -> Certificate -> IO SignedCertificate
signCertificate privKey cert = do
  let signFunction tbsData =
        case PKCS15.sign Nothing (Just SHA256) privKey tbsData of
          Left err -> error $ "RSA signing failed: " <> show err
          Right sig -> (sig, SignatureALG HashSHA256 PubKeyALG_RSA, ())
  let (signedCert, ()) = objectToSignedExact signFunction cert
  pure signedCert
```

#### Attestation Format

```json
{
  "url": "https://example.com/file.tar.gz",
  "host": "example.com",
  "sha256": "abcdef1234...",
  "size": 12345,
  "timestamp": "2026-01-25T14:30:00Z",
  "method": "GET",
  "cached": false
}
```

#### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_PORT` | 8080 | Listen port |
| `PROXY_CACHE_DIR` | /data/cache | Content cache |
| `PROXY_LOG_DIR` | /data/logs | Attestation logs |
| `PROXY_CERT_DIR` | /data/certs | CA certificates |
| `PROXY_ALLOWLIST` | (empty) | Allowed domains |

#### Build & Run

```bash
buck2 build //armitage/proxy:armitage-proxy

PROXY_PORT=8080 \
PROXY_CACHE_DIR=/var/cache/armitage \
PROXY_CERT_DIR=/etc/armitage/certs \
  ./result/bin/armitage-proxy
```

#### Dependencies

From the crypton ecosystem (maintained cryptonite fork):
- `tls` - TLS protocol
- `crypton-x509` - X.509 certificates
- `crypton` - Cryptographic primitives
- `pem`, `asn1-types`, `asn1-encoding` - Serialization

### Phase 1.5: NativeLink CAS Integration

**Status: Ready to Implement** - grapesy now builds on GHC 9.12

Integrate the witness proxy with NativeLink's content-addressed storage to unify
build artifact caching with fetch caching.

#### Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│                 │     │                      │     │                 │
│  Nix Build      │────▶│  Armitage Witness    │────▶│  Origin Server  │
│  (curl, wget)   │     │  Proxy               │     │                 │
│                 │◀────│                      │◀────│                 │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                               │
                               │ CAS API (gRPC/HTTP2)
                               ▼
                        ┌──────────────────┐
                        │  NativeLink CAS  │
                        │  (shared with    │
                        │   Buck2 RE)      │
                        └────────┬─────────┘
                                 │
                                 │ fast_slow store
                                 ▼
                        ┌──────────────────┐
                        │  Cloudflare R2   │
                        │  (S3-compatible) │
                        └──────────────────┘
```

#### Why NativeLink CAS?

1. **Unified storage** - Build outputs and fetched artifacts in same CAS
2. **Deduplication** - FastCDC chunking across all content
3. **Distributed** - R2 backend accessible from any region
4. **Already deployed** - Fly.io NativeLink infrastructure exists

#### grapesy on GHC 9.12

The `grapesy` gRPC library now builds on GHC 9.12 with our patches:

**Problem:** `proto-lens-setup` 0.4.0.9 used deprecated Cabal 3.14+ APIs:
- `matchDirFileGlob` changed to use `SymbolicPath` instead of `FilePath`
- `autogenComponentModulesDir` returns `SymbolicPath` instead of `FilePath`
- `extraSrcFiles` returns `[RelativePath Pkg File]` instead of `[FilePath]`

**Solution:** Patch file `nix/overlays/patches/proto-lens-setup-cabal-3.14.patch`:
- Imports `getSymbolicPath` and `makeSymbolicPath` under `MIN_VERSION_Cabal(3,14,0)`
- Converts between `SymbolicPath` and `FilePath` at API boundaries

**Haskell overlay** (`nix/overlays/haskell.nix`):
```nix
hs-pkgs = prev.haskell.packages.ghc912.override {
  overrides = hself: hsuper: {
    # ghc-source-gen from git (Hackage doesn't support GHC 9.12)
    ghc-source-gen = hself.callCabal2nix "ghc-source-gen" inputs.ghc-source-gen-src { };

    # proto-lens stack - jailbreak + patch
    proto-lens = doJailbreak hsuper.proto-lens;
    proto-lens-runtime = doJailbreak hsuper.proto-lens-runtime;
    proto-lens-protoc = doJailbreak hsuper.proto-lens-protoc;
    proto-lens-setup = appendPatch (doJailbreak hsuper.proto-lens-setup) proto-lens-setup-patch;
    proto-lens-protobuf-types = doJailbreak hsuper.proto-lens-protobuf-types;

    # grapesy stack - version pins
    http2 = hself.callHackageDirect { pkg = "http2"; ver = "5.3.9"; ... } {};
    http2-tls = hself.callHackageDirect { pkg = "http2-tls"; ver = "0.4.5"; ... } {};
    tls = hself.callHackageDirect { pkg = "tls"; ver = "2.1.4"; ... } {};
    grapesy = dontCheck (hself.callHackageDirect { pkg = "grapesy"; ver = "1.0.0"; ... } {});
  };
};
```

**Verification:**
```bash
$ nix develop -c ghc-pkg list | grep grapesy
    grapesy-1.0.0
    grpc-spec-1.0.0
    proto-lens-0.7.1.6

$ nix develop -c ghci -e ':m +Network.GRPC.Client' -e ':t withConnection'
withConnection :: ConnParams -> Server -> (Connection -> IO a) -> IO a
```

#### NativeLink Remote Execution API

NativeLink implements the [Remote Execution API](https://github.com/bazelbuild/remote-apis):

```protobuf
// ContentAddressableStorage service
service ContentAddressableStorage {
  // Find missing blobs before upload
  rpc FindMissingBlobs(FindMissingBlobsRequest) returns (FindMissingBlobsResponse);
  
  // Batch read small blobs
  rpc BatchReadBlobs(BatchReadBlobsRequest) returns (BatchReadBlobsResponse);
  
  // Batch upload small blobs
  rpc BatchUpdateBlobs(BatchUpdateBlobsRequest) returns (BatchUpdateBlobsResponse);
}

// ByteStream for large blobs
service ByteStream {
  rpc Read(ReadRequest) returns (stream ReadResponse);
  rpc Write(stream WriteRequest) returns (WriteResponse);
}

// Digest = (hash, size) tuple
message Digest {
  string hash = 1;        // SHA256 hex
  int64 size_bytes = 2;
}
```

#### CAS Client Module

New module `armitage/proxy/CAS.hs`:

```haskell
module Armitage.CAS
  ( CASClient
  , withCASClient
  , uploadBlob
  , downloadBlob
  , findMissingBlobs
  , Digest(..)
  ) where

import Network.GRPC.Client
import Network.GRPC.Client.StreamType.IO
import qualified Proto.Build.Bazel.Remote.Execution.V2.RemoteExecution as RE
import qualified Proto.Google.Bytestream as BS

-- | CAS client configuration
data CASConfig = CASConfig
  { casEndpoint :: String      -- ^ e.g., "localhost:50052" or "aleph-cas.fly.dev:443"
  , casUseTLS :: Bool          -- ^ Use TLS (required for Fly.io)
  , casInstanceName :: String  -- ^ RE instance name (usually "main")
  }

-- | Opaque CAS client handle
data CASClient = CASClient
  { casConn :: Connection
  , casConfig :: CASConfig
  }

-- | Create CAS client connection
withCASClient :: CASConfig -> (CASClient -> IO a) -> IO a
withCASClient config action = do
  let params = defaultConnParams
        { connTLS = if casUseTLS config then Just defaultTLSSettings else Nothing
        }
  withConnection params (Server (casEndpoint config)) $ \conn ->
    action (CASClient conn config)

-- | Upload blob to CAS (uses ByteStream.Write for large blobs)
uploadBlob :: CASClient -> Digest -> ByteString -> IO ()
uploadBlob client digest content
  | BS.length content < 4 * 1024 * 1024 = batchUpload client digest content
  | otherwise = streamUpload client digest content

-- | Download blob from CAS
downloadBlob :: CASClient -> Digest -> IO (Maybe ByteString)
downloadBlob client digest = do
  -- Use ByteStream.Read RPC
  let resourceName = casInstanceName (casConfig client) 
                  <> "/blobs/" <> digestHash digest 
                  <> "/" <> show (digestSize digest)
  streamDownload (casConn client) resourceName

-- | Check which blobs are missing
findMissingBlobs :: CASClient -> [Digest] -> IO [Digest]
findMissingBlobs client digests = do
  -- Use CAS.FindMissingBlobs RPC
  let request = RE.FindMissingBlobsRequest
        { instanceName = casInstanceName (casConfig client)
        , blobDigests = digests
        }
  response <- call (casConn client) (RPC @RE.ContentAddressableStorage @"findMissingBlobs") request
  pure (RE.missingBlobDigests response)
```

#### Witness Proxy Integration

Modify `armitage/proxy/Main.hs` to use CAS:

```haskell
-- Configuration
data ProxyConfig = ProxyConfig
  { proxyPort :: Int
  , proxyCacheDir :: FilePath     -- Local fallback
  , proxyLogDir :: FilePath
  , proxyCertDir :: FilePath
  , proxyAllowlist :: [String]
  , proxyCASEndpoint :: Maybe String  -- NEW: NativeLink CAS
  , proxyCASUseTLS :: Bool            -- NEW: TLS for CAS
  }

-- Cache response (CAS or local)
cacheResponse :: ProxyConfig -> ByteString -> IO ContentHash
cacheResponse config content = do
  let hash = sha256Hash content
  case proxyCASEndpoint config of
    Just endpoint -> do
      -- Upload to NativeLink CAS
      let casConfig = CASConfig endpoint (proxyCASUseTLS config) "main"
      withCASClient casConfig $ \client -> do
        let digest = Digest (hashToHex hash) (BS.length content)
        uploadBlob client digest content
    Nothing -> do
      -- Fall back to local filesystem
      let cachePath = proxyCacheDir config </> hashToHex hash
      BS.writeFile cachePath content
  pure hash

-- Check cache before fetching
checkCache :: ProxyConfig -> ContentHash -> IO (Maybe ByteString)
checkCache config hash = do
  case proxyCASEndpoint config of
    Just endpoint -> do
      let casConfig = CASConfig endpoint (proxyCASUseTLS config) "main"
      withCASClient casConfig $ \client -> do
        let digest = Digest (hashToHex hash) 0  -- Size unknown for lookup
        downloadBlob client digest
    Nothing -> do
      let cachePath = proxyCacheDir config </> hashToHex hash
      exists <- doesFileExist cachePath
      if exists then Just <$> BS.readFile cachePath else pure Nothing
```

#### NativeLink S3/R2 Store Configuration

NativeLink supports `experimental_s3_store` which works with R2:

```json
{
  "stores": [{
    "name": "FETCH_CACHE",
    "fast_slow": {
      "fast": {
        "filesystem": {
          "content_path": "/data/fetch-cache",
          "temp_path": "/data/fetch-temp",
          "eviction_policy": { "max_bytes": 10000000000 }
        }
      },
      "slow": {
        "experimental_s3_store": {
          "region": "auto",
          "bucket": "aleph-cas",
          "key_prefix": "fetches/",
          "endpoint_url": "https://<account>.r2.cloudflarestorage.com",
          "retry": {
            "max_retries": 6,
            "delay": 0.3,
            "jitter": 0.5
          }
        }
      }
    }
  }]
}
```

#### Benefits

1. **Single source of truth** - All content-addressed data in one place
2. **Cross-build sharing** - Fetch from one build reusable in another
3. **Global distribution** - R2 edge caching for worldwide access
4. **Attestation linkage** - Same hashes in build graph and fetch log

#### Implementation Steps

1. ✅ Port grapesy to GHC 9.12 (patch proto-lens-setup)
2. ✅ Add grapesy to devshell packages
3. ⬜ Create `armitage/proxy/CAS.hs` module
4. ⬜ Generate proto-lens bindings for Remote Execution API
5. ⬜ Add `--cas-endpoint` flag to proxy
6. ⬜ Configure NativeLink CAS with R2 slow store
7. ⬜ Deploy alongside existing Fly.io NativeLink
8. ⬜ Integration tests with local NativeLink

### Phase 1.6: Nix Binary Cache Facade

**Status: Planned**

Expose NativeLink CAS as a Nix binary cache, allowing `nix build` to substitute
store paths directly from our CAS infrastructure.

#### Nix Binary Cache Protocol

The protocol is simple HTTP:

```
GET /nix-cache-info
  → StoreDir: /nix/store
    WantMassQuery: 1
    Priority: 30

GET /<hash>.narinfo
  → StorePath: /nix/store/<hash>-<name>
    URL: nar/<filehash>.nar.zst
    Compression: zstd
    FileHash: sha256:<filehash>
    FileSize: 12345
    NarHash: sha256:<narhash>
    NarSize: 67890
    References: <dep1> <dep2> ...
    Sig: <key>:<signature>

GET /nar/<filehash>.nar.zst
  → (compressed NAR archive bytes)
```

#### Mapping to NativeLink CAS

```
┌─────────────────────────────────────────────────────────────────┐
│                     Nix Binary Cache Facade                      │
├─────────────────────────────────────────────────────────────────┤
│  GET /<hash>.narinfo     →  CAS lookup: narinfo/<hash>          │
│  GET /nar/<hash>.nar.zst →  CAS lookup: nar/<hash>              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ gRPC (ByteStream Read)
                              ▼
                    ┌──────────────────┐
                    │  NativeLink CAS  │
                    └──────────────────┘
```

The facade translates Nix's HTTP protocol to NativeLink's gRPC CAS API:

1. **narinfo files** - Stored in CAS under `narinfo/<storehash>` key
2. **NAR archives** - Stored in CAS under `nar/<filehash>` key
3. **Signing** - Facade signs narinfo on-the-fly with configured key

#### Implementation

Haskell HTTP server (warp) that speaks Nix binary cache protocol:

```haskell
-- Nix cache facade backed by NativeLink CAS
data NixCacheFacade = NixCacheFacade
  { casClient :: CASClient
  , signingKey :: Ed25519SecretKey
  , storeDir :: FilePath
  }

-- Handle narinfo request
handleNarinfo :: NixCacheFacade -> StoreHash -> IO (Maybe NarInfo)
handleNarinfo facade hash = do
  -- Fetch narinfo from CAS
  let digest = Digest { hash = "narinfo/" <> hash, size_bytes = 0 }
  result <- readBlob (casClient facade) "main" digest
  case result of
    Nothing -> pure Nothing
    Just bytes -> do
      let narinfo = parseNarInfo bytes
      -- Sign on the fly
      signed <- signNarInfo (signingKey facade) narinfo
      pure (Just signed)

-- Handle NAR request
handleNar :: NixCacheFacade -> FileHash -> IO (Maybe ByteString)
handleNar facade hash = do
  let digest = Digest { hash = "nar/" <> hash, size_bytes = 0 }
  readBlob (casClient facade) "main" digest
```

#### Populating the Cache

When builds complete (via Buck2 RE or direct), push outputs to CAS:

```bash
# After successful build
nix-store --dump /nix/store/<hash>-<name> | zstd > /tmp/out.nar.zst

# Upload NAR to CAS
cas-upload --key "nar/$(sha256sum /tmp/out.nar.zst)" /tmp/out.nar.zst

# Generate and upload narinfo
nix path-info --json /nix/store/<hash>-<name> | \
  generate-narinfo | \
  cas-upload --key "narinfo/<hash>"
```

#### Benefits

1. **Unified infrastructure** - Same CAS for builds and Nix substitution
2. **Global distribution** - R2 backend means worldwide edge caching
3. **No separate cache service** - NativeLink already deployed
4. **Attestation integration** - Can require signed attestations for substitution

#### Trust Model

The facade signs narinfo files. Nix clients configure:

```nix
{
  nix.settings = {
    substituters = [ "https://cache.straylight.cx" ];
    trusted-public-keys = [ "cache.straylight.cx:ABC123..." ];
  };
}
```

Only builds with valid attestations get their narinfo signed, enforcing
the coeffect discharge requirement at the cache level.

#### Available Haskell Packages (GHC 9.12)

| Package | Version | Status | Use |
|---------|---------|--------|-----|
| `grapesy` | 1.0.0 | ✓ builds (patched) | gRPC client/server |
| `grpc-spec` | 1.0.0 | ✓ builds | gRPC protocol types |
| `proto-lens` | 0.7.1.6 | ✓ builds (jailbreak) | Protocol buffer codegen |
| `http2` | 5.3.9 | ✓ builds | HTTP/2 support |
| `http2-tls` | 0.4.5 | ✓ builds | HTTP/2 over TLS |
| `tls` | 2.1.4 | ✓ builds | TLS protocol |
| `warp` | 3.4.x | ✓ builds | HTTP server |
| `hnix-store-core` | 0.8.0.0 | ✓ builds (crypton) | Hashes, signatures, store paths |
| `hnix-store-nar` | 0.1.1.0 | ✗ (cryptonite) | NAR serialization |
| `hnix-store-remote` | 0.7.0.0 | ✗ (cryptonite) | Daemon protocol client |
| `nix-derivation` | 1.1.3 | ✓ builds | .drv file parsing |

**All phases unblocked:**

- **Phase 1.5 (NativeLink CAS)**: grapesy works on GHC 9.12
- **Phase 1.6 (Nix cache facade)**: warp + hnix-store-core available

**Patches applied** (in `nix/overlays/haskell.nix`):
- `ghc-source-gen` from git (Hackage version doesn't support GHC 9.12)
- `proto-lens-setup` patched for Cabal 3.14+ SymbolicPath API
- `proto-lens-*` jailbroken for base 4.21 / ghc-prim 0.13

With `hnix-store-core` 0.8.0.0, we get:
- `System.Nix.Hash` - Nix hashing (SHA256, base32 encoding)
- `System.Nix.Signature` - NAR signature verification/creation  
- `System.Nix.StorePath` - Store path parsing
- `System.Nix.Fingerprint` - Fingerprinting for signing

### Phase 2: Dhall Build Schema (Weeks 3-4)

```dhall
-- Build.dhall
let Resource = ./Resource.dhall
let Toolchain = ./Toolchain.dhall

let Target = {
  name : Text,
  srcs : List Text,
  deps : List Text,
  toolchain : Toolchain,
  requires : Resource
}

let target : Target → Target = λ(t : Target) → t
```

### Phase 3: Coeffect Checker (Weeks 5-6)

Lean4 validates that coeffects can be discharged:

```lean
def checkBuild (b : Build) (env : Environment) : Except Error Unit := do
  let required := b.requires
  let available := env.resources
  if canDischarge required available
    then pure ()
    else throw (CoefficientError required available)
```

### Phase 4: Gentry → Armitage Migration

Replace the current prelude chaos with armitage:

```nix
# Old (gentry prototype)
aleph.stdenv.clang-glibc-static.mkDerivation { ... }

# New (armitage)  
armitage.build {
  src = armitage.fetch "sha256:abc";  # R2, not github
  toolchain = armitage.toolchain.clang-18-glibc-static;
  requires = armitage.resource.network;
}
```

## Naming

| Component | Name | Role |
|-----------|------|------|
| Compatibility shim | **Armitage** | Routes typed builds through daemon |
| Microvm executor | **isospin** | Firecracker fork for isolated builds |
| GPU multiplexer | **hypercharge** | Userspace nvidia.ko broker |
| Toolchain | **Toolchain** | Standard (compiler × host × target × flags) |
| Target triple | **Triple** | Standard (arch-vendor-os-abi) |
| Resource algebra | **Resource** | Coeffects (what builds require) |
| Signed proof | **Attestation** | Coeffect discharge evidence |

## Phase 5: Graded Monad Execution

**Status: Design**

The build system executes graded monad computations with Armitage as the effect
handler. CA derivations only - input-addressed derivations pass through without
attestation.

### The Execution Model

A build is not a shell script. It's a **computation** with typed effects:

```lean
-- A build is a graded monad indexed by its effects
def myBuild : Build [Fetch, Write] OutputHash := do
  let src ← fetch "https://example.com/src.tar.gz"
  let compiled ← exec "gcc" ["-o", "out", "main.c"]
  write compiled
```

Armitage **interprets** this computation:

```
┌─────────────────────────────────────────────────────────────────┐
│  Build Spec (Lean/Dhall)                                        │
│  myBuild : Build [Fetch, Write] OutputHash                      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Armitage Executor                                              │
│  - Runs computation step by step                                │
│  - Intercepts each effect                                       │
│  - Records execution trace                                      │
│  - Proxies fetches through witness                              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Lean Verifier                                                  │
│  - Checks: trace ⊢ spec                                         │
│  - Discharges proof obligation                                  │
│  - Signs (spec, trace, output)                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Signed Attestation                                             │
│  { spec, trace, output, proof, signature }                      │
└─────────────────────────────────────────────────────────────────┘
```

### CA vs Input-Addressed

The attestation model only applies to **content-addressed** derivations:

| Derivation Type | Armitage Behavior |
|-----------------|-------------------|
| CA (`__contentAddressed = true`) | Full attestation: proxy, trace, sign |
| Input-addressed (legacy) | Pass through: `nix-store --realise`, no attestation |

For CA derivations, the output hash is derived from content. Witnessing fetches
gives cryptographic proof of what went in. For input-addressed, the hash is from
inputs, so attestation is meaningless.

```bash
#!/bin/sh
# Minimal CA-aware executor

DRV=$1

if nix show-derivation "$DRV" | jq -e '.[].env.__contentAddressed == "1"' >/dev/null; then
  # CA: full attestation mode
  HTTP_PROXY=http://127.0.0.1:8888 \
  SSL_CERT_FILE=/etc/ssl/armitage/ca.pem \
  nix-store --realise "$DRV"
  # Attestation log now has proof
else
  # Legacy: just build it
  nix-store --realise "$DRV"
fi
```

### The Proxy as Substitutor

The witness proxy does double duty:

1. **Witness mode**: Intercept fetches during build, log attestations
2. **Substitute mode**: Serve cached content by CA hash, verify attestations

```
┌─────────────────────────────────────────────────────────────────┐
│  Armitage Proxy                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  fetch("https://...") ──┬──→ cache hit? return + attest         │
│                         │                                       │
│                         └──→ cache miss? fetch, store,          │
│                              attest, return                     │
│                                                                 │
│  substitute("/nix/store/xxx") ──→ lookup by CA in bucket        │
│                                   verify attestation            │
│                                   return blob                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The cache IS the attestation bucket. No separate binary cache needed.

### Bucket Layout

```dhall
-- Armitage/Storage.dhall

let HashType = < SHA256 | BLAKE3 >

let ContentAddress = 
  { hash : Text
  , hashType : HashType 
  }

let BucketLayout =
  { specs : Text        -- s3://bucket/specs/{hash}.lean
  , traces : Text       -- s3://bucket/traces/{hash}.cbor
  , outputs : Text      -- s3://bucket/cas/{hash}
  , proofs : Text       -- s3://bucket/proofs/{hash}.olean
  , attestations : Text -- s3://bucket/attestations/{hash}.json
  }

let defaultLayout : BucketLayout =
  { specs = "specs/"
  , traces = "traces/"
  , outputs = "cas/"
  , proofs = "proofs/"
  , attestations = "attestations/"
  }
```

Storage hierarchy:

```
s3://aleph-continuity/
├── specs/              # Build computations (Lean terms)
│   └── {hash}.lean
├── traces/             # Execution traces (CBOR)
│   └── {hash}.cbor
├── cas/                # Content-addressed blobs
│   └── {hash}
├── proofs/             # Compiled Lean proofs
│   └── {hash}.olean
└── attestations/       # Signed attestations
    └── {hash}.json
```

### Attestation Schema

```dhall
-- Armitage/Attestation.dhall

let Effect = 
  < Fetch : { url : Text, hash : ContentAddress }
  | Write : { path : Text, hash : ContentAddress }
  | Exec : { cmd : Text, args : List Text, exitCode : Natural }
  >

let Trace =
  { computation : ContentAddress  -- hash of spec
  , effects : List Effect         -- what actually happened
  , startTime : Text              -- ISO 8601
  , endTime : Text
  , executor : Text               -- machine identity
  }

let SignedAttestation =
  { spec : ContentAddress         -- hash of Build computation
  , trace : ContentAddress        -- hash of execution trace
  , output : ContentAddress       -- hash of build output
  , proofTerm : Optional Text     -- serialized Lean proof (if verified)
  , signature : Text              -- ed25519 over (spec, trace, output, proof)
  , signingKey : Text             -- public key
  , timestamp : Text              -- ISO 8601
  }
```

### Lean Proof Structure

The proof discharged by verification:

```lean
-- Armitage/Proof.lean

structure ContentHash where
  bytes : ByteArray
  deriving Repr, BEq, Hashable

structure Fetch where
  url : String
  hash : ContentHash
  
structure Trace where
  spec : ContentHash
  fetches : List Fetch
  output : ContentHash

-- The core theorem: CA derivations are reproducible
-- If two builds fetch identical content, they produce identical output
theorem ca_reproducible 
    (t₁ t₂ : Trace) 
    (hspec : t₁.spec = t₂.spec) 
    (hfetch : t₁.fetches = t₂.fetches) : 
    t₁.output = t₂.output := by
  -- Discharged by the witness log
  -- The attestation IS the proof witness
  sorry

-- Verification: trace satisfies spec
def verifyTrace (spec : Spec) (trace : Trace) : Bool :=
  -- Check each effect in trace matches spec
  -- Check hashes are consistent
  -- Check no disallowed effects occurred
  sorry
```

The signature is over the **discharged proof** - not just "I saw these bytes"
but "I verified this trace satisfies this computation."

### Substitutor Protocol

When Nix asks for a store path:

```
nix build
  │
  ├── needs /nix/store/abc-foo
  │
  ▼
Armitage Substitutor
  │
  ├── lookup: attestations/abc.json
  │     └── verify signature against trusted keys
  │
  ├── if trusted:
  │     └── serve: cas/{output-hash}
  │
  └── if untrusted:
        └── 404 (force rebuild with attestation)
```

Configuration:

```dhall
-- Armitage/Substitutor.dhall

let TrustPolicy =
  < TrustKey : Text                    -- trust single key
  | TrustAny : List Text               -- trust any of these keys
  | TrustAll : List Text               -- require all keys
  | TrustThreshold : { keys : List Text, n : Natural }  -- n-of-m
  >

let SubstitutorConfig =
  { bucket : Text
  , trustPolicy : TrustPolicy
  , fallbackToBuild : Bool             -- rebuild if no trusted attestation
  }
```

### Integration with isospin

The armitage-builder VM runs this model:

1. VM boots with nimi + armitage proxy
2. Receives `.drv` to build
3. If CA: proxy all fetches, record trace, verify, sign
4. If legacy: just `nix-store --realise`
5. Push attestation + output to bucket
6. Shutdown

```
┌────────────────────────────────────────────────────────────────┐
│  isospin VM                                                    │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  nimi (PID 1)                                            │  │
│  │    └── armitage-proxy service (:8888)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Environment:                                                  │
│    HTTP_PROXY=http://127.0.0.1:8888                           │
│    SSL_CERT_FILE=/etc/ssl/armitage/ca.pem                     │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Build Process                                           │  │
│  │    nix-store --realise /nix/store/xxx.drv               │  │
│  │    └── all fetches → armitage → attested                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Output:                                                       │
│    /var/log/armitage/trace.cbor    → bucket/traces/           │
│    /nix/store/result               → bucket/cas/              │
│    attestation.json                → bucket/attestations/     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Phase 6: DNS and Sovereign Naming

**Status: Design**

The naming system uses DNS as a **cache** for content-addressed truth, not as
the source of truth. The canonical domain is `straylight.cx`.

### DNS Records

```
; Attestation records - TXT with structured data
_att.llvm.straylight.cx.        TXT "v=att1 ca=sha256:abc... key=ed25519:xyz..."
_att.llvm.18.straylight.cx.     TXT "v=att1 ca=sha256:def... key=ed25519:xyz..."

; Trust anchors - the signing keys
_trust.straylight.cx.           TXT "v=trust1 key=ed25519:xyz... exp=2027-01-01"

; CAS endpoints - where to fetch content  
_cas.straylight.cx.             TXT "v=cas1 grpc=cas.straylight.cx:443 s3=r2.straylight.cx"

; Git attestation repo
_git.straylight.cx.             TXT "v=git1 repo=git.straylight.cx/attestations"
```

### Dynamic DNS Server

The DNS server **computes** responses from the attestation store:

```
┌─────────────────────────────────────────────────────────────────┐
│  ns1.straylight.cx                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query: _att.llvm.18.straylight.cx TXT                         │
│                                                                 │
│  1. Parse query → (package: "llvm", version: "18")             │
│  2. Lookup in attestation store (git/NativeLink)               │
│  3. Find latest attestation matching (llvm, 18, trusted key)   │
│  4. Return TXT record with CA hash + signature                 │
│                                                                 │
│  No zone files. Attestation store IS the zone.                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Sovereign Name Format

```dhall
-- Armitage/Name.dhall

let Scheme = < CA | Att | Git | Nix | Spec >

let Authority =
  < DNS : Text                   -- domain name
  | Key : Text                   -- ed25519 public key (self-sovereign)
  | None                         -- content-addressed, no authority
  >

let SovereignName =
  { scheme : Scheme
  , authority : Authority
  , path : List Text             -- package / subpath
  , version : Optional Text
  , trustPolicy : Optional TrustPolicy
  , contentAddress : Optional Hash
  }
```

### URI Schemes

```
ca://sha256:abcdef1234...                     # exact content (fully sovereign)
att://straylight.cx/llvm@18                   # attested by trusted key
git://sha256:repo-hash@main/llvm              # git ref, repo by hash
spec://sha256:build-term-hash                 # "run this computation"
nix://github:nixos/nixpkgs#hello              # legacy compat
```

The `ca://` scheme is **fully sovereign** - no DNS, no authority, just math.
The `att://` scheme uses DNS as a **convenience layer** but the signature chain
bottoms out in content hashes.

### Nix Registry Integration

Grandfather `nix run` via registry override:

```json
{
  "version": 2,
  "flakes": [
    {
      "from": { "type": "indirect", "id": "nixpkgs" },
      "to": { 
        "type": "tarball", 
        "url": "https://resolve.straylight.cx/nixpkgs"
      }
    }
  ]
}
```

The resolver service translates names to CAS redirects:

```
nix run nixpkgs#hello
    │
    ▼
registry: nixpkgs → resolve.straylight.cx/nixpkgs
    │
    ▼
resolve.straylight.cx:
    ├── DNS: _att.nixpkgs.straylight.cx TXT → ca=sha256:abc...
    ├── verify signature
    └── redirect → r2.straylight.cx/cas/sha256:abc...
```

### Infrastructure

| Service | Domain | Purpose |
|---------|--------|---------|
| DNS | ns1.straylight.cx | Dynamic DNS from attestation store |
| Resolver | resolve.straylight.cx | Name → CAS redirect |
| CAS | cas.straylight.cx | NativeLink gRPC endpoint |
| Storage | r2.straylight.cx | Cloudflare R2 (S3-compatible) |
| Git | git.straylight.cx | Attestation repo |

### Hash Format

Canonical form for cross-system compatibility:

```dhall
-- Armitage/Hash.dhall

let Algorithm = < SHA256 | BLAKE3 | SHA1 >

let Hash = 
  { algorithm : Algorithm
  , bytes : Text              -- hex encoded
  , size : Natural            -- content size (RE API needs this)
  }

-- Canonical: algo:hex
-- sha256:abcdef1234...
-- blake3:abcdef1234...
let render : Hash -> Text = 
  \(h : Hash) -> 
    merge 
      { SHA256 = "sha256:", BLAKE3 = "blake3:", SHA1 = "sha1:" } 
      h.algorithm 
    ++ h.bytes
```

## References

- Petricek et al., "Coeffects: A calculus of context-dependent computation"
- Orchard et al., "Quantitative Type Theory"
- Gaboardi et al., "Linear Dependent Types for Differential Privacy"
