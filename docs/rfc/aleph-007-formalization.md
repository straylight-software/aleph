# ℵ-007: Nix Formalization

| Field | Value |
|-------|-------|
| RFC | ℵ-007 |
| Title | Nix Formalization |
| Author | b7r6 |
| Status | In Progress |
| Created | 2026-01-17 |

## Abstract

This RFC describes the ongoing work to formalize Nix's build semantics through
three interconnected components:

1. **Content-Addressed Derivations** - Always-on CA derivations in straylight-nix
1. **Typed Package DSL** - Haskell-defined packages compiled to WASM, evaluated via `builtins.wasm`
1. **Isospin Builder** - Firecracker-based instant-boot VM builder runtime

## Motivation

Nix's current implementation has several areas where behavior is implicit,
underspecified, or dependent on runtime configuration:

- Derivation hashing depends on whether CA derivations is enabled
- Package definitions are untyped Nix expressions
- Build isolation relies on Linux namespaces (leaky) or macOS sandbox (incomplete)
- Repair, remote building, and caching have different code paths for CA vs non-CA

Formalizing these layers gives us:

- **Predictable hashing**: Derivations compute their hash the same way everywhere
- **Type safety**: Package definitions are validated at compile time
- **True isolation**: VM-based builds with clean slate every time
- **Verifiable builds**: Content-addressed outputs enable proof of reproducibility

## Component 1: Content-Addressed Derivations

### What CA Derivations Changes

When `ca-derivations` is enabled, Nix tracks **realisations** - mappings from
derivation outputs to their actual store paths. This enables:

- Derivations can be renamed without changing their output hash
- Early cutoff: if a rebuilt output is identical, dependents don't rebuild
- Output paths are determined by content, not by derivation hash

### Always-On Implementation

In `straylight-nix`, we make CA derivations always enabled in
`src/libutil/configuration.cc`:

```cpp
bool ExperimentalFeatureSettings::isEnabled(const ExperimentalFeature & feature) const
{
    if (feature == Xp::CaDerivations ||      // Content-addressed derivations
        feature == Xp::PipeOperators ||      // Pure syntax sugar
        feature == Xp::FetchTree ||          // Required by flakes
        feature == Xp::FetchClosure ||       // Safe, enables better caching
        feature == Xp::ParseTomlTimestamps)  // TOML spec compliance
        return true;
    // ...
}
```

### Test Failures and Root Causes

Enabling CA derivations by default exposed four test failures:

#### 1. `repair` Test

**Symptom**: Corrupted derivation not repaired by `nix-store --verify --repair`

**Root Cause**: In `derivation-building-goal.cc:checkPathValidity()`, when CA
derivations is enabled and a realisation exists, the code was unconditionally
setting `PathStatus::Valid` without checking if the actual path contents match
the expected hash:

```cpp
// BUG: Was always setting Valid when realisation exists
if (auto real = worker.store.queryRealisation(drvOutput)) {
    info.known = {
        .path = real->outPath,
        .status = PathStatus::Valid,  // <-- Should check content hash!
    };
}
```

This caused repair mode to skip rebuilding because the path was considered
"already valid" even though the contents were corrupted.

**Fix**: Check path validity the same way as non-CA code path:

```cpp
if (auto real = worker.store.queryRealisation(drvOutput)) {
    auto outputPath = real->outPath;
    info.known = {
        .path = outputPath,
        .status = !worker.store.isValidPath(outputPath)               ? PathStatus::Absent
                  : !checkHash || worker.pathContentsGood(outputPath) ? PathStatus::Valid
                                                                      : PathStatus::Corrupt,
    };
}
```

**Location**: `src/libstore/build/derivation-building-goal.cc:1136-1144`

#### 2. `structured-attrs` Test

**Symptom**: `exportReferencesGraph` fails with "cannot export references of
path '...' because it is not in the input closure"

**Root Cause**: In `nix-shell`, when setting up the shell environment for a
derivation with structured attrs, `prepareStructuredAttrs` is called. This
function calls `exportReferences`, which requires the exported paths to be in
the input closure. However, `nix-shell` only computed the input closure from
`drv.inputDrvs`, not including paths from `exportReferencesGraph`.

**Location**: `src/nix/nix-build/nix-build.cc:581-600`

**Fix**: Add paths from `exportReferencesGraph` to the inputs before calling
`prepareStructuredAttrs`:

```cpp
for (const auto & [inputDrv, inputNode] : drv.inputDrvs.map)
    accumInputClosure(inputDrv, inputNode);

/* Also add paths from exportReferencesGraph to inputs, since
   prepareStructuredAttrs will call exportReferences which requires
   these paths to be in the input closure. */
for (const auto & [_, storePaths] : drvOptions.exportReferencesGraph)
    for (const auto & p : storePaths)
        store->computeFSClosure(p, inputs);

auto json = drv.structuredAttrs->prepareStructuredAttrs(*store, drvOptions, inputs, drv.outputs);
```

#### 3. `check` Test

**Symptom**: Determinism check behavior differs with CA derivations

**Root Cause**: The `--check` flag rebuilds and compares outputs. With CA
derivations, the comparison logic needs to account for realisation-based
output paths rather than derivation-based paths.

#### 4. `build-remote-content-addressed-fixed` Test

**Symptom**: Build log not available for fixed-output CA derivations built
via `ssh-ng://` protocol on remote machines.

**Root Cause**: The build hook (`nix __build-remote`) has unsound pipe/fd
handling in the log streaming path. The architecture involves:

1. Parent process (derivation-building-goal) creates pipes to hook
1. Hook (build-remote) spawns remote build via `ssh://` or `ssh-ng://`
1. For `ssh-ng://`, daemon sends `STDERR_RESULT` messages with log lines
1. build-remote's `processStderr()` forwards these via JSONLogger to stderr
1. Parent reads JSON from `hook->fromHook` pipe and writes to log file

The bug is in step 4-5: JSON messages with `action: "result"` carry the
**daemon's** activity ID, but `handleJSONLogMessage` looks up activities in
the **local** `hook->activities` map. The lookup fails silently, and while
the code at `derivation-building-goal.cc:1043-1047` attempts to write log
lines to the sink regardless, the pipe/fd handling is fundamentally unsound.

For `ssh://` (LegacySSHStore), build output goes directly through
`hook->builderOut` which bypasses this JSON dance entirely - hence it works.

**Resolution**: Remote builders disabled entirely. The build hook is a
Superfund site of pipe juggling that would require a rewrite to fix properly.
CA derivations work correctly for local builds - that's the important part.

```cpp
Setting<Strings> buildHook{
    this,
    {}, // Remote builders disabled - the build hook implementation is unsound
    "build-hook",
    // ...
};
```

Users can re-enable with `build-hook = nix __build-remote` at their own risk.

### Resolution Status

| Test | Issue | Status |
|------|-------|--------|
| `repair` | Realisation path not hash-checked | **Fixed** |
| `structured-attrs` | exportReferencesGraph paths not in inputs | **Fixed** |
| `check` | Same as repair | **Fixed** |
| `build-remote-*` | Build hook unsound | **Disabled** |

### Fixes Applied

1. **Repair/Check Fix** (`src/libstore/build/derivation-building-goal.cc`):
   Check content hash when realisation exists, not just assume valid.

1. **nix-shell Fix** (`src/nix/nix-build/nix-build.cc`): Include
   `exportReferencesGraph` paths in the input set for structured attrs.

1. **Test Update** (`tests/functional/ca/derivation-json.sh`): Remove test
   for "feature disabled" error since CA is always enabled.

## Component 2: Typed Package DSL

### The Core Abstraction

Packages can be written in any language that compiles to WASM. The prelude's
`call-package` handles this transparently - the file extension determines the
backend:

```nix
# In an overlay - this is how you define packages
final: prev: {
  nvidia-sdk = final.call-package ./nvidia-sdk.hs {};
  zlib-ng = final.call-package ./zlib-ng.hs {};
  fmt = final.call-package ./fmt.purs {};    # PureScript works too
  hello = final.call-package ./hello.nix {}; # So does Nix
}
```

There is no special namespace. Packages are just packages. The typed source
file is just another way to write a derivation, like `.nix`.

### How `call-package` Works

```
┌──────────────────────────────────────────────────────────────────┐
│  Source File (*.hs, *.purs, *.nix)                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼ call-package detects extension
┌──────────────────────────────────────────────────────────────────┐
│  Backend Selection                                               │
│  .hs   → GHC WASM compiler → WASM derivation                     │
│  .purs → PureScript WASM   → WASM derivation                     │
│  .nix  → Native Nix import → Nix attrset                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼ builtins.wasm (for .hs/.purs)
┌──────────────────────────────────────────────────────────────────┐
│  Package Spec (attrset)                                          │
│  { pname, version, src, builder, deps, phases, meta, ... }       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼ build-from-spec
┌──────────────────────────────────────────────────────────────────┐
│  Derivation                                                      │
└──────────────────────────────────────────────────────────────────┘
```

The WASM compilation is cached and content-addressed. If the source file hasn't
changed, the WASM derivation is reused from the store.

### The Type Universe

The Haskell/PureScript side uses a unified type system:

```haskell
Drv                    -- A derivation (pure data)
Sh a                   -- A script action (effectful)
Drv -> Drv             -- A package transformation
Sh Drv                 -- A script that produces a derivation
```

A builder is just a script. A script can reference packages. A package can
have a script as its builder.

### Flake Module Configuration

The typed package system is configured via a flake module:

```nix
{
  aleph.build = {
    enable = true;
    
    # Language backends
    languages = {
      haskell.enable = true;
      purescript.enable = true;
      nix.enable = true;  # Validate .nix against typed schema
    };
  };
}
```

The module:

1. Provides `call-package` in the prelude overlay
1. Configures which language backends are available
1. Sets up the WASM compilation toolchain

### Haskell Bindings

Located in `nix/scripts/Aleph/Nix/`:

- `FFI.hs` - Raw C imports for WASM FFI
- `Value.hs` - High-level NixValue API
- `Derivation.hs` - Core `Drv` type and `ToNixValue` serialization
- `Syntax.hs` - Nix-like DSL (`mkDerivation`, `fetchurl`, etc.)

### Package DSL

The syntax mirrors Nix but is fully typed:

```haskell
module Aleph.Nix.Packages.Nvidia where

nccl :: Drv
nccl = mkDerivation
    [ pname "nvidia-nccl"
    , version "2.28.9"
    , src $ fetchurl
        [ url "https://pypi.nvidia.com/nvidia-nccl-cu13/..."
        , hash "sha256-5FU6MPNBlfP6HaAqbaPWM30o8gA5Q6oKPSR7vCX+/EI="
        ]
    , dontUnpack True
    , nativeBuildInputs ["autoPatchelfHook", "unzip"]
    , buildInputs ["stdenv.cc.cc.lib", "zlib"]
    , installPhase
        [ unzip "unpacked"
        , mkdir "lib"
        , mkdir "include"
        , copy "unpacked/nvidia/nccl/lib/." "lib/"
        , copy "unpacked/nvidia/nccl/include/." "include/"
        , symlink "lib" "lib64"
        ]
    , description "NVIDIA NCCL 2.28.9 (from PyPI)"
    , homepage "https://developer.nvidia.com/nccl"
    , license "unfree"
    ]
```

### Typed Build Actions

Build phases use typed actions, not bash strings:

```haskell
data Action
    = WriteFile Text Text           -- path, content
    | Install Int Text Text         -- mode, src, dst
    | Mkdir Text                    -- path (relative to $out)
    | Symlink Text Text             -- target, link
    | Copy Text Text                -- src, dst
    | Remove Text                   -- path
    | Unzip Text                    -- dest dir (extracts $src)
    | PatchElfRpath Text [Text]     -- binary, rpaths
    | PatchElfAddRpath Text [Text]  -- binary, rpaths to add
    | Substitute Text [(Text,Text)] -- file, [(from, to)]
    | Wrap Text [WrapAction]        -- program, wrapper actions
    | Run Text [Text]               -- escape hatch: cmd, args

data WrapAction
    = WrapPrefix Text Text     -- --prefix VAR : value
    | WrapSuffix Text Text     -- --suffix VAR : value
    | WrapSet Text Text        -- --set VAR value
    | WrapSetDefault Text Text -- --set-default VAR value
    | WrapUnset Text           -- --unset VAR
    | WrapAddFlags Text        -- --add-flags "flags"
```

The Nix-side interpreter converts these to safe, properly-quoted shell commands.
No string interpolation bugs. No quoting errors.

### Scripts as Builders

When custom build logic is needed, the builder is a typed script:

```haskell
myProject :: Drv
myProject = mkDerivation
    [ pname "my-project"
    , version "1.0"
    , src $ github "me/myproject" "v1.0"
    , buildInputs ["zlib", "openssl"]
    , builder $ script $ do
        configure
        make
        install [bin "my-tool", lib "libmy.so"]
    ]
```

The `script` function embeds a `Sh ()` in the derivation. At build time,
this runs in the sandbox with full type safety.

### Multi-Language Support

The WASM ABI is the contract, not the source language. PureScript can use
the same DSL:

```purescript
module Aleph.Nix.Packages.Nvidia where

nccl :: Drv
nccl = mkDerivation
    [ pname "nvidia-nccl"
    , version "2.28.9"
    -- ... same structure
    ]
```

Both Haskell and PureScript compile to WASM. Both produce identical Nix
attrsets. The host language is a choice of ergonomics.

### Current Implementation Status

| Component | Status |
|-----------|--------|
| `builtins.wasm` in straylight-nix | **Complete** |
| GHC WASM toolchain | **Complete** |
| `Aleph.Nix.{FFI,Value,Derivation,Syntax}` | **Complete** |
| Typed actions (all common patterns) | **Complete** |
| NVIDIA packages via WASM | **Complete** (6 packages) |
| `call-package` for .hs files | **Complete** |
| Zero-boilerplate wrapper Main.hs | **Complete** |
| `tool()` for auto dependency tracking | **Complete** |
| Typed tool modules (Jq, PatchElf, Install, Substitute) | **Complete** |
| Incremental adoption bridge | **Next** |
| PureScript backend | **Planned** |
| Flake module configuration | **Planned** |

### Example: Complete Package File

A single `.hs` file defines a package. No registry, no exports, no boilerplate:

```haskell
-- nix/packages/nvidia-sdk.hs
module Main where

import Aleph.Nix.Syntax

main :: IO ()
main = export $ mkDerivation
    [ pname "nvidia-nccl"
    , version "2.28.9"
    , src $ fetchurl
        [ url "https://pypi.nvidia.com/nvidia-nccl-cu13/..."
        , hash "sha256-..."
        ]
    , dontUnpack True
    , nativeBuildInputs ["autoPatchelfHook", "unzip"]
    , buildInputs ["stdenv.cc.cc.lib", "zlib"]
    , installPhase
        [ unzip "unpacked"
        , copy "unpacked/nvidia/nccl/lib/." "lib/"
        , symlink "lib" "lib64"
        ]
    ]
```

Used in an overlay:

```nix
final: prev: {
  nvidia-sdk.nccl = final.call-package ./nix/packages/nvidia-sdk.hs {};
}
```

The file *is* the package. `call-package` compiles it, evaluates it, builds it.

## Component 3: Isospin Builder

*Documentation to be added as implementation progresses.*

Isospin is a Firecracker-derived instant-boot VM runtime for Nix builds:

- **Instant boot**: ~125ms VM startup time
- **True isolation**: Hardware virtualization, not namespaces
- **Clean slate**: Fresh VM state every build
- **Reproducible**: Deterministic VM configuration

### Integration Points

- Replaces `src/libstore/unix/build/` sandbox implementation
- Uses virtio-fs or virtio-blk for store access
- Communicates with Nix daemon via vsock

## Migration Path

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | CA derivations always-on, remote builders disabled | **Complete** |
| 2 | Core typed DSL (Drv, Action, FFI, Syntax) | **Complete** |
| 3 | `call-package` for .hs files | **Complete** |
| 4 | Zero-boilerplate wrapper (auto Main.hs generation) | **Complete** |
| 5 | `tool()` for automatic dependency tracking | **Complete** |
| 6 | Typed tool modules (Jq, PatchElf, Install, Substitute) | **Complete** |
| 7 | Incremental adoption (`aleph.phases.interpret`) | **Next** |
| 8 | More typed tools (Wrap, Chmod, CMake, Meson) | Planned |
| 9 | Cross-compilation support (`buildPackages` refs) | Planned |
| 10 | Multiple outputs support | Planned |
| 11 | Flake module configuration for language backends | Planned |
| 12 | PureScript backend with shared WASM ABI | Planned |
| 13 | Isospin builder prototype | Planned |
| 14 | Disable legacy escape hatches | Planned |

### Phase 7: Incremental Adoption (Current Priority)

The key blocker to "default everywhere" is migration friction. Existing packages
can't be rewritten overnight. The solution is a bridge:

```nix
# Traditional package with one typed phase
stdenv.mkDerivation {
  pname = "existing-package";
  # ... all existing attrs unchanged ...
  
  # Just replace postInstall with typed actions
  postInstall = aleph.phases.interpret [
    (Jq.query { rawOutput = true; } ".version" "$out/package.json")
    (PatchElf.setRpath "bin/myapp" [ (PatchElf.rpathOut "lib") ])
  ];
}
```

This allows:

- Gradual migration, one phase at a time
- No "big bang" rewrites
- Typed benefits immediately for the phases that need it
- Existing CI/tooling continues to work

Note: Phase 7 (Isospin) will provide remote building capability with a clean
implementation. The current build hook is not worth fixing - it's easier to
replace it with a VM-based builder that has proper isolation and logging.

### Disabling Legacy Patterns

Once `call-package` handles typed sources, the overlay can enforce migration:

```nix
# Eventually, in the prelude overlay
final: prev: {
  writeShellApplication = throw "Use a .hs file with Aleph.Script";
  writeShellScript = throw "Use a .hs file with Aleph.Script";
  
  # stdenv.mkDerivation still works but warns on string phases
  # (implementation TBD)
}
```

The goal: **zero bash, zero untyped strings**. The typed file is the
specification. Nix is just the build orchestrator.

## Appendix: Typed Tool Modules

Typed tool modules live in `nix/scripts/Aleph/Nix/Tools/`. Each provides a
type-safe interface to a common build tool, with automatic dependency tracking.

### Jq (`Aleph.Nix.Tools.Jq`)

```haskell
import qualified Aleph.Nix.Tools.Jq as Jq

-- Query with options
Jq.query Jq.defaults { Jq.rawOutput = True } ".version" "package.json"

-- Query multiple files
Jq.queryFiles Jq.defaults ".name" ["a.json", "b.json"]

-- Transform in-place
Jq.transform Jq.defaults ".version = \"2.0\"" "package.json"
```

### PatchElf (`Aleph.Nix.Tools.PatchElf`)

```haskell
import qualified Aleph.Nix.Tools.PatchElf as PatchElf

-- Set rpath with typed entries
PatchElf.setRpath "bin/myapp"
    [ PatchElf.rpathOut "lib"           -- $out/lib
    , PatchElf.rpathPkg "zlib" "lib"    -- ${zlib}/lib
    , PatchElf.rpathOrigin "/../lib"    -- $ORIGIN/../lib
    ]

-- Add to existing rpath
PatchElf.addRpath "lib/libfoo.so" [PatchElf.rpathPkg "openssl" "lib"]

-- Set interpreter
PatchElf.setInterpreter "bin/myapp" "${glibc}/lib/ld-linux-x86-64.so.2"

-- Shrink unused entries
PatchElf.shrinkRpath "bin/myapp"
```

### Install (`Aleph.Nix.Tools.Install`)

```haskell
import qualified Aleph.Nix.Tools.Install as Install

-- Type-safe modes (no chmod bugs)
Install.bin "build/myapp" "bin/myapp"       -- mode 755
Install.lib "libfoo.so" "lib/libfoo.so"     -- mode 644
Install.header "foo.h" "include/foo.h"      -- mode 644
Install.data_ "icon.png" "share/app/icon.png"
Install.doc "README.md" "share/doc/app/README.md"
Install.man "myapp.1" "share/man/man1/myapp.1"

-- Explicit mode
Install.file 0o600 "secrets.conf" "etc/app/secrets.conf"

-- Directory operations
Install.dir "share/myapp/data"
Install.tree "doc/html" "share/doc/myapp/html"
```

### Substitute (`Aleph.Nix.Tools.Substitute`)

```haskell
import qualified Aleph.Nix.Tools.Substitute as Sub

-- substituteInPlace with --replace-fail (safe default)
Sub.inPlace "Makefile"
    [ Sub.replace "@PREFIX@" "$out"
    , Sub.replace "@VERSION@" "1.0.0"
    ]

-- Regex variant
Sub.inPlaceRegex "config.h"
    [ Sub.replaceRegex "#define DEBUG [01]" "#define DEBUG 0"
    ]
```

### Adding New Tools

To add a typed tool module:

1. Create `nix/scripts/Aleph/Nix/Tools/MyTool.hs`
1. Define pure data types for options
1. Create functions returning `Action` values
1. Use `ToolRun (PkgRef "mytool") [args]` for auto-deps
1. Add to `extraModules` in `wasm-plugin.nix`
1. Re-export from `Aleph.Nix.Tools`

The tool is automatically added to `nativeBuildInputs` when used.

## References

- [Nix RFC 62: Content-Addressed Derivations](https://github.com/NixOS/rfcs/pull/62)
- [GHC WASM Backend](https://ghc.gitlab.haskell.org/ghc/doc/users_guide/wasm.html)
- [Firecracker MicroVM](https://firecracker-microvm.github.io/)
- [straylight-nix wasm-flat branch](https://github.com/DeterminateSystems/nix/tree/wasm-flat)
