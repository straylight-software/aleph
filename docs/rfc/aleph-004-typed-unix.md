# ℵ-004: Aleph.Script — Typed Shell Infrastructure

| Field | Value |
|-------|-------|
| RFC | ℵ-004 |
| Title | Aleph.Script — Typed Shell Infrastructure |
| Author | Straylight |
| Status | Draft |
| Created | 2025-01-17 |

## Abstract

This RFC specifies **Aleph.Script**, a typed shell scripting infrastructure that replaces bash
scripts with compiled Haskell while preserving the ergonomics of shell programming. The system
provides:

1. Type-safe wrappers for CLI tools (generated from `--help` output)
1. Domain modules for common operations (OCI containers, VFIO, VMs)
1. A Dhall-based configuration bridge for Nix → Haskell value passing
1. ~2ms startup time (compiled, not interpreted)

## Motivation

### The Problem

The Nix ecosystem is drowning in untyped shell code. Not just scripts — the *build phases themselves*.

Open any package in nixpkgs. Here's what you'll find:

```nix
stdenv.mkDerivation {
  # ...
  
  postPatch = ''
    substituteInPlace src/config.h \
      --replace '/usr/local' "$out" \
      --replace 'VERSION' '"${version}"'
    
    sed -i 's|/bin/bash|${bash}/bin/bash|g' scripts/*.sh
    
    for f in $(find . -name '*.py'); do
      substituteInPlace "$f" \
        --replace "#!/usr/bin/env python" "#!${python}/bin/python"
    done
  '';
  
  buildPhase = ''
    runHook preBuild
    
    export CFLAGS="$CFLAGS -I${openssl.dev}/include"
    export LDFLAGS="$LDFLAGS -L${openssl.out}/lib"
    
    make -j$NIX_BUILD_CORES PREFIX=$out \
      CC="${stdenv.cc}/bin/cc" \
      ${lib.optionalString stdenv.isDarwin "MACOS=1"} \
      ${lib.optionalString enableFeature "FEATURE=1"}
    
    runHook postBuild
  '';
  
  installPhase = ''
    runHook preInstall
    
    mkdir -p $out/bin $out/lib $out/share/man/man1
    
    install -m755 build/tool $out/bin/
    install -m644 build/*.so $out/lib/
    
    # Generate config
    cat > $out/etc/config.json << EOF
    {
      "prefix": "$out",
      "version": "${version}",
      "features": ${builtins.toJSON features}
    }
    EOF
    
    wrapProgram $out/bin/tool \
      --prefix PATH : ${lib.makeBinPath [ coreutils jq ]} \
      --set SSL_CERT_FILE "${cacert}/etc/ssl/certs/ca-bundle.crt"
    
    runHook postInstall
  '';
}
```

This is **normal**. This is **everywhere**. This is 80,000+ packages.

Count the bugs:

1. **Unquoted `$f`** in the for loop — breaks on filenames with spaces
1. **Heredoc inside Nix string** — `EOF` doesn't work, you need `'EOF'`, but then interpolation breaks
1. **`${version}` vs `"${version}"`** — one is Nix, one is shell, which is which?
1. **`$NIX_BUILD_CORES`** — unquoted, could be empty
1. **`$CFLAGS`** — word splitting intended here, but `$out` word splitting is not
1. **Missing error handling** — `make` fails silently if `CC` path is wrong
1. **`$(find ...)`** — command substitution splits on whitespace
1. **`lib.optionalString`** — produces empty string, not omitted argument

And this is a *simple* package. Real packages have:

```nix
postInstall = ''
  ${lib.optionalString stdenv.isLinux ''
    # Now we're in Nix interpolation inside a Nix string
    for lib in $out/lib/*.so; do
      patchelf --set-rpath "${lib.makeLibraryPath [ openssl zlib ]}" "$lib"
    done
    
    ${lib.optionalString enableSystemd ''
      # Nested interpolation inside interpolation
      install -Dm644 ${./systemd.service} $out/lib/systemd/system/${pname}.service
      substituteInPlace $out/lib/systemd/system/${pname}.service \
        --replace '@out@' "$out" \
        --replace '@path@' "${lib.makeBinPath [ coreutils ]}"
    ''}
  ''}
  
  ${lib.optionalString stdenv.isDarwin ''
    # Different code path, different bugs
    install_name_tool -change /usr/lib/libSystem.B.dylib \
      ${darwin.Libsystem}/lib/libSystem.B.dylib \
      $out/bin/tool
  ''}
'';
```

The `${}` means different things in different contexts:

- At the outer level: Nix interpolation
- Inside `''`: still Nix interpolation
- Inside `''` after a shell variable: shell parameter expansion
- Inside heredocs: depends on quoting
- Inside `lib.optionalString`: Nix interpolation, then shell expansion

**This is why we have a "Forbidden Patterns" document that starts with "ZERO HEREDOCS ARE PERMITTED ON PAIN OF DEATH."**

But the heredoc ban only treats the symptom. The disease is **untyped string manipulation of shell code**.

`writeShellApplication` is actually the *best* case — it at least runs shellcheck and sets `set -euo pipefail`. But it's still:

```nix
pkgs.writeShellApplication {
  name = "deploy";
  text = ''
    crane export --plaform linux/amd64 "$IMAGE"  # typo: plaform
    bwrap --bind $ROOTFS /                        # unquoted variable
    tar -xf - -C "$OUTPUT"                        # error lost in pipe
  '';
}
```

1. **Typos in flags silently fail** — `--plaform` isn't `--platform`, shellcheck can't know
1. **Unquoted variables** — shellcheck catches some, but not semantic errors
1. **Errors lost in pipes** — `set -o pipefail` helps but doesn't compose
1. **No structure** — everything is strings, no paths vs text vs commands
1. **No completion** — your editor doesn't know crane's flags

### The Solution

Compile Haskell scripts that:

- Catch typos at compile time (`Crane.plaform_` is an error)
- Handle arguments as typed data (`[Text]`, not string interpolation)
- Compose error handling (`ExceptT`, `Either`, explicit failure modes)
- Distinguish paths, commands, and text at the type level

```haskell
-- This won't compile if you typo a flag
Crane.export_ [ Crane.platform_ "linux/amd64" ] image
  |> Tar.extract output
```

## Design

### Layer 1: Aleph.Script (The Shell)

A Shelly-based foundation that feels like bash but isn't:

```haskell
import Aleph.Script

main :: IO ()
main = script $ do
  -- Like bash, but typed
  echo ":: Starting deployment"
  
  files <- ls "."
  forM_ files $ \f -> do
    when (hasExtension f ".hs") $ do
      run "ghc" ["-O2", toTextIgnore f]
  
  -- Errors are explicit
  result <- try $ run "might-fail" []
  case result of
    Left err -> echoErr $ "Failed: " <> tshow err
    Right _  -> echo "Success"
```

Key exports from `Aleph.Script`:

| Function | Type | Bash Equivalent |
|----------|------|-----------------|
| `echo` | `Text -> Sh ()` | `echo` |
| `echoErr` | `Text -> Sh ()` | `echo >&2` |
| `run` | `FilePath -> [Text] -> Sh Text` | `command args` |
| `run_` | `FilePath -> [Text] -> Sh ()` | `command args` (ignore output) |
| `ls` | `FilePath -> Sh [FilePath]` | `ls` |
| `cp` | `FilePath -> FilePath -> Sh ()` | `cp` |
| `mv` | `FilePath -> FilePath -> Sh ()` | `mv` |
| `rm` | `FilePath -> Sh ()` | `rm` |
| `mkdir_p` | `FilePath -> Sh ()` | `mkdir -p` |
| `cd` | `FilePath -> Sh ()` | `cd` |
| `pwd` | `Sh FilePath` | `pwd` |
| `which` | `FilePath -> Sh (Maybe FilePath)` | `which` |
| `exit` | `Int -> Sh a` | `exit` |
| `try` | `Sh a -> Sh (Either SomeException a)` | `if command; then` |
| `catch` | `Sh a -> (SomeException -> Sh a) -> Sh a` | `trap` |

### Layer 2: Tool Wrappers (Generated)

Auto-generated from `--help` output. Two parser backends:

**Clap (Rust tools)**: `rg`, `fd`, `bat`, `delta`, `dust`, `tokei`, `hyperfine`, etc.

```haskell
import qualified Aleph.Script.Tools.Rg as Rg

-- Type-safe ripgrep
results <- Rg.run
  [ Rg.pattern_ "TODO"
  , Rg.glob_ "*.hs"
  , Rg.context_ 3
  , Rg.json_           -- output as JSON
  ] ["."]
```

**GNU getopt_long**: `ls`, `grep`, `sed`, `find`, `tar`, `rsync`, etc.

```haskell
import qualified Aleph.Script.Tools.Tar as Tar

-- Type-safe tar
Tar.run_
  [ Tar.extract_
  , Tar.file_ archive
  , Tar.directory_ output
  , Tar.verbose_
  ] []
```

**Hand-crafted domain wrappers**: `jq`, `crane`, `bwrap`

```haskell
import qualified Aleph.Script.Tools.Bwrap as Bwrap

-- Builder pattern for complex tools
let sandbox = Bwrap.new
      |> Bwrap.bind rootfs "/"
      |> Bwrap.dev "/dev"
      |> Bwrap.proc "/proc"
      |> Bwrap.tmpfs "/tmp"
      |> Bwrap.roBind "/etc/resolv.conf" "/etc/resolv.conf"
      |> Bwrap.setenv "PATH" "/usr/bin:/bin"
      |> Bwrap.unshare [Bwrap.Pid, Bwrap.Net]
      |> Bwrap.dieWithParent

Bwrap.exec sandbox ["/bin/bash"]
```

### Layer 3: Domain Modules

Higher-level operations composed from tool wrappers:

**Aleph.Script.Oci** — Container image operations

```haskell
import qualified Aleph.Script.Oci as Oci

-- Pull or use cache, returns rootfs path
rootfs <- Oci.pullOrCache Oci.defaultConfig "alpine:latest"

-- Build a sandbox configuration
let sandbox = Oci.baseSandbox rootfs
      |> Oci.withGpu          -- Add GPU device bindings
      |> Oci.withNetwork      -- Keep network namespace

Oci.exec sandbox ["/bin/sh"]
```

**Aleph.Script.Vfio** — GPU passthrough

```haskell
import qualified Aleph.Script.Vfio as Vfio

-- Bind GPU and all IOMMU group members to vfio-pci
devices <- Vfio.bindToVfio "0000:01:00.0"

-- Later: unbind and rescan
Vfio.unbindFromVfio "0000:01:00.0"
```

**Aleph.Script.Vm** — Virtual machine operations

```haskell
import qualified Aleph.Script.Vm as Vm

-- Build a rootfs from an OCI image
rootfs <- Vm.buildRootfs Vm.defaultConfig image

-- Launch with Firecracker
Vm.launchFirecracker kernel rootfs config
```

### Layer 4: Dhall Configuration Bridge

The critical problem: Nix computes values (paths, config) that Haskell needs at runtime.

#### The Type Tower

Start with one type that catches 99% of bugs:

```dhall
-- Aleph/Config/Base.dhall

-- A validated Nix store path
-- The string MUST start with /nix/store/
-- Nix guarantees this invariant when generating configs
let StorePath = Text

in { StorePath }
```

This single type catches:

- Typos in derivation references
- Wrong attribute paths
- Missing dependencies

#### Haskell Side

```haskell
-- Aleph.Script/Config.hs

-- Newtype prevents mixing store paths with arbitrary strings
newtype StorePath = StorePath { unStorePath :: Text }
  deriving (Generic, FromDhall, Show, Eq)

-- Smart constructor for runtime validation (defense in depth)
mkStorePath :: Text -> Either Text StorePath
mkStorePath t
  | "/nix/store/" `T.isPrefixOf` t = Right (StorePath t)
  | otherwise = Left $ "Invalid store path: " <> t

-- Convert to FilePath for use with Shelly
toFilePath :: StorePath -> FilePath
toFilePath = fromText . unStorePath
```

#### Nix Side

```nix
# Generate Dhall config with Nix interpolation
mkDhallConfig = name: expr: pkgs.writeText "${name}.dhall" expr;

isospin-run-config = mkDhallConfig "isospin-run" ''
  { kernel = "${isospin-kernel-pkg}"
  , initScript = "${isospin-run-init}"
  , busybox = "${pkgs.pkgsStatic.busybox}/bin/busybox"
  , sslCerts = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
  , cpus = ${toString cfg.firecracker.cpus}
  , memMib = ${toString cfg.firecracker.mem-mib}
  , platform = "${oci-platform}"
  }
'';
```

#### Consumption

**Option A: Compile-time embed (zero runtime cost)**

```haskell
{-# LANGUAGE TemplateHaskell #-}

import Dhall.TH

config :: FcRunConfig
config = $(staticDhallExpression "./config.dhall")
```

**Option B: Runtime load (flexible for development)**

```haskell
main :: IO ()
main = do
  configPath <- fromMaybe "./config.dhall" <$> lookupEnv "WEYL_CONFIG"
  config <- inputFile auto configPath
  script $ runWithConfig config
```

#### Progressive Refinement

The type tower grows as patterns stabilize:

```
Phase 1: StorePath (catches 99% of bugs)
    ↓
Phase 2: OciConfig, VmConfig (domain bundles)
    ↓
Phase 3: Kernel, Rootfs, Sandbox (precise types)
    ↓
Phase 4: Validated formats, enums, constraints
```

```dhall
-- Phase 2: Domain configs
let OciConfig = {
  platform : Text,
  cacheDir : Text,
  crane : StorePath,
  bwrap : StorePath,
  sslCerts : StorePath
}

let VmConfig = {
  cpus : Natural,
  memMib : Natural,
  kernel : StorePath,
  hugepages : Bool
}

-- Phase 3: Precise types (when stable)
let KernelFormat = < Vmlinux | Bzimage | Efi >

let Kernel = {
  path : StorePath,
  format : KernelFormat,
  bootArgs : Text
}
```

## Implementation

### File Structure

```
nix/scripts/
├── Aleph/
│   ├── Script.hs                 # Main module (Shelly re-exports)
│   ├── Config/
│   │   ├── Base.dhall            # StorePath, primitives
│   │   ├── Oci.dhall             # OCI-specific types
│   │   ├── Vm.dhall              # VM-specific types
│   │   └── Types.hs              # Haskell FromDhall instances
│   └── Script/
│       ├── Clap.hs               # Clap --help parser
│       ├── Getopt.hs             # GNU getopt_long parser
│       ├── Oci.hs                # OCI operations
│       ├── Vfio.hs               # VFIO/PCI binding
│       ├── Vm.hs                 # VM rootfs/config
│       └── Tools/
│           ├── Bwrap.hs          # bubblewrap (hand-crafted)
│           ├── Crane.hs          # crane (hand-crafted)
│           ├── Jq.hs             # jq (hand-crafted)
│           ├── Rg.hs             # ripgrep (generated)
│           ├── Fd.hs             # fd (generated)
│           └── ...               # 24 more tool wrappers
├── vfio-bind.hs                  # Compiled scripts
├── unshare-run.hs
├── isospin-run.hs
└── ...
```

### Nix Integration

```nix
# nix/overlays/script.nix

mkCompiledScript = {
  name,
  deps ? [],
  configFile ? null,  # Optional Dhall config
}:
pkgs.stdenv.mkDerivation {
  inherit name;
  src = scriptSrc;
  
  nativeBuildInputs = [ ghcWithDeps pkgs.makeWrapper ];
  
  buildPhase = ''
    ${lib.optionalString (configFile != null) ''
      cp ${configFile} Config.dhall
    ''}
    ghc -O2 -Wall -hidir . -odir . -i$src -o ${name} $src/${name}.hs
  '';
  
  installPhase = ''
    mkdir -p $out/bin
    cp ${name} $out/bin/
  '';
  
  postFixup = lib.optionalString (deps != []) ''
    wrapProgram $out/bin/${name} --prefix PATH : ${lib.makeBinPath deps}
  '';
};

# Usage
aleph.script.compiled = {
  vfio-bind = mkCompiledScript {
    name = "vfio-bind";
    deps = [ pkgs.pciutils ];
  };
  
  isospin-run = mkCompiledScript {
    name = "isospin-run";
    deps = [ pkgs.firecracker pkgs.crane ];
    configFile = isospin-run-config;
  };
};
```

### Wrapper Generation

```bash
# Generate a wrapper from --help output
nix run .#aleph.script.gen-wrapper -- rg

# Force GNU format
nix run .#aleph.script.gen-wrapper -- grep --gnu

# Write directly to Tools/
nix run .#aleph.script.gen-wrapper -- fd --write
```

The generator:

1. Runs `tool --help` and captures output
1. Detects format (Clap vs GNU) from structure
1. Parses flags, options, and arguments
1. Generates a Haskell module with typed constructors
1. Includes documentation from help text

## Comparison

| Aspect | Bash | Aleph.Script |
|--------|------|-------------|
| Type safety | None | Full (compile-time) |
| Startup time | ~2ms | ~2ms (compiled) |
| Error handling | `set -e` (fragile) | `Either`/`ExceptT` (composable) |
| Argument passing | String interpolation | `[Text]` (no injection) |
| Flag typos | Silent failure | Compile error |
| IDE support | Limited | Full (HLS) |
| Debugging | `set -x` | GHC profiling, stack traces |
| Dependencies | Runtime `PATH` | Wrapped at build time |

## Migration Path

### Phase 1: Tool Wrappers (Complete)

27 tool wrappers generated and tested:

- Clap: rg, fd, bat, delta, dust, tokei, hyperfine, deadnix, statix, stylua, taplo, zoxide
- GNU: ls, grep, sed, find, xargs, tar, gzip, wget, rsync
- Hand-crafted: jq, crane, bwrap

### Phase 2: Container Scripts (Complete)

13 scripts converted from bash to Haskell:

- VFIO: vfio-bind, vfio-unbind, vfio-list
- OCI: unshare-run, unshare-gpu, crane-inspect, crane-pull
- Namespace: fhs-run, gpu-run
- Firecracker: isospin-run, isospin-build
- Cloud Hypervisor: cloud-hypervisor-run, cloud-hypervisor-gpu

### Phase 3: Dhall Config Bridge (In Progress)

- Define `Base.dhall` with `StorePath`
- Add `FromDhall` instances
- Update `mkCompiledScript` for config injection
- Convert isospin-run as proof of concept

### Phase 4: Remaining Scripts

Census of `writeShellApplication` across aleph:

- `lint.nix`: lint-init, lint-link (converted)
- `container/`: all scripts (converted)
- `script.nix`: gen-wrapper, check, props (meta-scripts, keep)

## Why Dhall

Dhall is the right choice for the config bridge:

| Aspect | Raw Haskell Gen | JSON | Dhall |
|--------|-----------------|------|-------|
| Type safety | Manual | None | Automatic |
| Schema location | Duplicated | External | Single source |
| Tooling | None | jq | dhall format/lint/diff |
| Error messages | Nix eval | Runtime | Dhall type errors |
| Imports | None | None | URLs, files, inline |
| Normalization | None | None | Canonical form |

Dhall catches errors at config generation time:

```dhall
-- This fails if you put a non-store-path in a StorePath field
{ kernel = "not-a-store-path" }  -- Type error!

-- This works
{ kernel = "/nix/store/abc123-linux/vmlinux" }
```

## The Broader Context

### nixpkgs is 80,000 Packages of Untyped Shell

This isn't just a aleph problem. It's the Nix ecosystem's original sin.

Every `mkDerivation` in nixpkgs has:

- `configurePhase` — bash
- `buildPhase` — bash
- `checkPhase` — bash
- `installPhase` — bash
- `fixupPhase` — bash
- `preConfigurePhases`, `postInstallPhases`, ... — all bash

The phases are concatenated strings with Nix interpolation. The result is a bash script
that runs in a sandbox. There is no type checking. There is no completion. There is
barely any tooling.

**stdenv.mkDerivation is fundamentally a bash code generator.**

This is why:

- Half of nixpkgs PRs are fixing quoting bugs
- `substituteInPlace` has 17 different failure modes
- Cross-compilation is fragile (host vs build vs target shell)
- Overriding packages often breaks in subtle ways

### The Cost

Every Nix user has experienced:

1. **Build failures with cryptic messages** — `make: *** No targets specified` because
   a variable was empty due to quoting
1. **Works locally, fails in CI** — different `PATH`, different shell
1. **Overrides that silently do nothing** — the phase string was already evaluated
1. **Hours debugging string interpolation** — is it `${}` or `''${}` or `\${}` or `'''\${}`?

### Why We Can't Fix nixpkgs

nixpkgs is too big to rewrite. 80,000 packages with bash phases are not going to be
converted to anything else. The community has too much invested in the current model.

But we can stop the bleeding for *new* code:

- New scripts in aleph use Aleph.Script
- Build phases remain bash (for now) but are minimal wrappers
- Complex logic moves to compiled Haskell called from phases

### The Long Game

If Aleph.Script proves itself, the pattern can spread:

1. **aleph scripts** — fully typed (this RFC)
1. **aleph build phases** — minimal bash calling compiled tools
1. **Upstream contributions** — offer typed alternatives for common patterns
1. **New build systems** — Buck2/Bazel integration doesn't use bash phases at all

The goal isn't to boil the ocean. It's to make typed infrastructure the path of least
resistance for new code, until the bash legacy is isolated to backward compatibility.

## Conformance

Code is **Aleph.Script conformant** if it:

1. Uses `Aleph.Script` for all shell operations (no raw `System.Process`)
1. Uses generated tool wrappers for CLI tools (no `run "rg" [...]`)
1. Passes Nix values via Dhall configs (no hardcoded paths)
1. Compiles with `-Wall -Werror`
1. Has ~2ms startup time (no interpreted Haskell)

## Part II: Unified Package and Script DSL

### The Unification

Phase 1-4 established typed shell scripting. The next step is recognizing that **packages and scripts are the same thing**:

- A `Drv` is a pure value describing what to build
- A `Sh a` is an effectful computation that runs commands
- A builder is just a script that runs at build time

```haskell
-- These are the same type universe
Drv                    -- A derivation (pure data)
Sh a                   -- A script action (effectful)
Drv -> Drv             -- A package transformation
Sh Drv                 -- A script that produces a derivation
FilePath -> Sh ()      -- A script that takes a path
```

### The Interface

From Nix, it's just `call-package` with a typed source file:

```nix
# In an overlay
final: prev: {
  nvidia-sdk = {
    nccl = final.call-package ./nvidia/nccl.hs {};
    cudnn = final.call-package ./nvidia/cudnn.hs {};
  };
  
  zlib-ng = final.call-package ./zlib-ng.hs {};
  fmt = final.call-package ./fmt.purs {};  # PureScript works too
}
```

No special namespace. No registry. The file extension determines the backend:

- `.hs` → GHC WASM → `builtins.wasm` → spec → derivation
- `.purs` → PureScript WASM → `builtins.wasm` → spec → derivation
- `.nix` → Native Nix import (validated against schema)

### Package Definition

Packages use the same DSL as scripts, just with pure data:

```haskell
module Aleph.Packages.Nvidia where

import Aleph.Nix.Syntax

-- Version pins (pure data)
ncclVersion = "2.28.9"
cudnnVersion = "9.17.0.29"

-- Package definition (pure data)
nccl :: Drv
nccl = wheel "nvidia-nccl" ncclVersion $ do
  from "https://pypi.nvidia.com/nvidia-nccl-cu13/..."
  sha256 "5FU6MPNBlfP6HaAqbaPWM30o8gA5Q6oKPSR7vCX+/EI="
  extract "nvidia/nccl/lib" ~> lib
  extract "nvidia/nccl/include" ~> include
  link [gcc.lib, zlib]
```

The `wheel` function and its monadic DSL (`from`, `sha256`, `extract`, `link`) produce a `Drv` value. No bash. No untyped strings.

### Builder Scripts

When a package needs custom build logic, the builder is a script:

```haskell
module Aleph.Packages.MyProject where

import Aleph.Script
import Aleph.Nix.Syntax

myProject :: Drv
myProject = mkDrv "my-project" "1.0" $ do
  src $ github "me/myproject" "v1.0"
  deps [zlib, openssl]
  
  -- The builder is a Sh () that runs at build time
  builder $ do
    configure
    make
    install [bin "my-tool", lib "libmy.so"]
```

The `builder` function takes a `Sh ()` and embeds it in the derivation. At build time, this script runs in the sandbox.

### Scripts That Use Packages

Scripts can reference packages naturally:

```haskell
module Aleph.Scripts.Deploy where

import Aleph.Script
import qualified Aleph.Packages.Nvidia as Nvidia

-- Script that uses NCCL
withNccl :: Sh a -> Sh a
withNccl = withPackage Nvidia.nccl

-- Script that builds with NCCL
buildWithNccl :: FilePath -> Sh ()
buildWithNccl srcDir = withNccl $ do
  cmake srcDir $ do
    flag "NCCL_ROOT" (packagePath Nvidia.nccl)
    buildType Release
  make
```

### Package Transformations

Overrides are typed functions:

```haskell
-- Override a package
withDebug :: Drv -> Drv
withDebug = addFlags ["-g", "-O0"]

-- Apply it
ncclDebug :: Drv  
ncclDebug = withDebug Nvidia.nccl

-- Compose transformations
ncclCustom :: Drv
ncclCustom = Nvidia.nccl
  |> withDebug
  |> withDeps [extraLib]
  |> withPatches [./fix.patch]
```

### The Compilation Model

```
┌─────────────────────────────────────────────────────────────────┐
│  Haskell/PureScript Source Tree                                 │
│  (modules, packages, scripts - all typed)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ GHC WASM / PureScript
┌─────────────────────────────────────────────────────────────────┐
│  WASM Module (cached, content-addressed)                        │
│  Exports: eval(module, name, args) -> NixValue                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ builtins.wasm
┌─────────────────────────────────────────────────────────────────┐
│  Nix Evaluation                                                 │
│  aleph.eval "Module.name" { args }                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Nix Value (derivation, attrset, function, etc.)                │
└─────────────────────────────────────────────────────────────────┘
```

Each typed source file compiles to its own WASM module, cached and content-
addressed. `call-package` handles the compilation transparently - the user
just writes `.hs` files and uses them like any other package source.

### Multi-Language Support

The core insight: **the WASM ABI is the contract, not the source language.**

PureScript can compile the same semantic DSL:

```purescript
-- PureScript: same semantics, different syntax
module Aleph.Packages.Nvidia where

nccl :: Drv
nccl = wheel "nvidia-nccl" "2.28.9" do
  from "https://pypi.nvidia.com/nvidia-nccl-cu13/..."
  sha256 "5FU6MPNBlfP6HaAqbaPWM30o8gA5Q6oKPSR7vCX+/EI="
  extract "nvidia/nccl/lib" ~> lib
  extract "nvidia/nccl/include" ~> include
  link [gcc.lib, zlib]
```

Both compile to WASM. Both produce the same Nix attrset shape. The host language is a choice of ergonomics, not semantics.

### Why This Is Better

| Aspect | Current (Bash in Nix) | Unified Typed DSL |
|--------|----------------------|-------------------|
| Type safety | None | Full (compile-time) |
| Package definition | Nix attrset + bash strings | Haskell/PureScript types |
| Build phases | Bash fragments | Typed `Sh ()` actions |
| Overrides | `overrideAttrs` (stringly) | Typed functions |
| Composition | String concatenation | Function composition |
| IDE support | None | Full (HLS/purs-ide) |
| Error messages | Runtime bash failures | Compile-time type errors |
| Documentation | Comments in strings | Haddock/docstrings |

### The Path Forward

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core typed DSL (Drv, Action, FFI) | **Complete** |
| 2 | WASM compilation + builtins.wasm | **Complete** |
| 3 | call-package for .hs files | **Complete** |
| 4 | Auto-wrapper Main.hs (zero boilerplate) | **Complete** |
| 5 | tool() for auto dependency tracking | **Complete** |
| 6 | Typed tool modules (Jq, PatchElf, Install, Substitute) | **Complete** |
| 7 | Incremental adoption (aleph.phases.interpret) | **Next** |
| 8 | More typed tools (Wrap, Chmod, CMake, Meson) | Planned |
| 9 | PureScript backend with shared WASM ABI | Planned |
| 10 | Cross-compilation support | Planned |
| 11 | Multiple outputs support | Planned |
| 12 | Disable legacy patterns | Planned |

The goal: **zero bash, zero untyped strings, zero heredocs**. The typed file
is the specification. Nix is just the build orchestrator.

## Part III: Roadmap to Default Everywhere

### Current State (as of 2026-01-19)

The typed package system is **functional** but not yet the **default**:

```
┌─────────────────────────────────────────────────────────────────┐
│  What Works Today                                                │
├─────────────────────────────────────────────────────────────────┤
│  • call-package ./my-package.hs {} → derivation                  │
│  • Typed actions: Install, Substitute, PatchElf, Jq             │
│  • Auto tool deps: use jq → jq added to nativeBuildInputs       │
│  • Zero boilerplate: module Pkg where; pkg = mkDerivation [...]  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  What's Missing                                                  │
├─────────────────────────────────────────────────────────────────┤
│  • Incremental adoption for existing packages                    │
│  • More typed tool modules                                       │
│  • Cross-compilation (buildPackages references)                  │
│  • Multiple outputs support                                      │
│  • IDE integration for .hs package files                         │
└─────────────────────────────────────────────────────────────────┘
```

### Gap 1: Incremental Adoption

**Problem**: Existing packages use `mkDerivation` with bash phases. Rewriting
them all at once is impractical.

**Solution**: `aleph.phases.interpret` — a bridge that lets traditional Nix
packages use typed phases without full migration:

```nix
stdenv.mkDerivation {
  pname = "my-existing-package";
  # ... all existing attrs ...
  
  # Replace just the postInstall phase with typed actions
  postInstall = aleph.phases.interpret [
    (Jq.query Jq.defaults { rawOutput = true; } ".version" "$out/package.json")
    (PatchElf.setRpath "bin/myapp" [ (PatchElf.rpathOut "lib") ])
  ];
}
```

Implementation:

1. `aleph.phases.interpret` takes a list of typed actions
1. Converts them to shell script via `actionsToShell`
1. Returns the shell string for use in traditional phases
1. Tool deps are extracted and must be added manually (for now)

**Status**: Not implemented. Priority: High.

### Gap 2: Tool Coverage

**Current typed tools:**

- `Aleph.Nix.Tools.Jq` — jq queries
- `Aleph.Nix.Tools.PatchElf` — rpath manipulation
- `Aleph.Nix.Tools.Install` — typed install with correct modes
- `Aleph.Nix.Tools.Substitute` — substituteInPlace

**Needed for real packages:**

| Tool | Priority | Notes |
|------|----------|-------|
| `Wrap` | High | wrapProgram with typed env mods (have Action, need module) |
| `Chmod` | High | mode changes (trivial to add) |
| `CMake` | High | typed configure options |
| `Meson` | Medium | typed meson options |
| `Cargo` | Medium | Rust build configuration |
| `autoPatchelfHook` | Medium | automatic rpath fixing |
| `find` + bulk ops | Medium | typed file discovery |
| `sed` | Low | prefer Substitute, but sometimes needed |
| `ln` | Low | have Symlink action, may need module |

**Implementation approach**: Add tools as needed by real package migrations.
Don't speculatively build tools — let demand drive it.

### Gap 3: Interpreter Completeness

The Nix-side interpreter (`actionToShell` in `wasm-plugin.nix`) handles:

```nix
actionToShell = action:
  if action.action == "writeFile" then ...
  else if action.action == "install" then ...
  else if action.action == "mkdir" then ...
  else if action.action == "symlink" then ...
  else if action.action == "copy" then ...
  else if action.action == "remove" then ...
  else if action.action == "unzip" then ...
  else if action.action == "patchelfRpath" then ...
  else if action.action == "patchelfAddRpath" then ...
  else if action.action == "substitute" then ...
  else if action.action == "wrap" then ...
  else if action.action == "run" then ...
  else if action.action == "toolRun" then ...
  else throw "Unknown action type";
```

**Missing**: No known gaps. New actions require:

1. Add constructor to `Action` in `Derivation.hs`
1. Add serialization in `actionToNix`
1. Add interpreter clause in `actionToShell`

### Gap 4: Cross-Compilation

Typed packages need to reference `buildPackages` for native tools:

```haskell
-- Current: works but can't distinguish host vs build
nativeBuildInputs ["cmake", "pkg-config"]

-- Needed: explicit buildPackages reference
nativeBuildInputs
    [ buildPackages "cmake"
    , buildPackages "pkg-config"
    ]
```

Implementation:

1. Add `BuildPkgRef` to `PkgRef` type
1. Update dependency resolution in `buildFromSpec`
1. Add DSL helper: `buildPkg :: Text -> Text`

**Status**: Not implemented. Priority: Medium (needed for cross-compilation).

### Gap 5: Multiple Outputs

Standard packages often have multiple outputs:

```haskell
-- Current: single output only
pkg = mkDerivation [ pname "foo", ... ]

-- Needed: multiple outputs
pkg = mkDerivation
    [ pname "foo"
    , outputs ["out", "dev", "lib"]
    , installPhase
        [ Install.header "include/foo.h" "dev:include/foo.h"
        , Install.lib "libfoo.so" "lib:lib/libfoo.so"
        , Install.bin "foo" "out:bin/foo"
        ]
    ]
```

Implementation:

1. Add `outputs :: [Text] -> DrvAttr` to Syntax
1. Extend `OutPath` to include output name
1. Update interpreter to handle `$dev`, `$lib`, etc.
1. Add `outPath' :: Text -> Text -> OutPath` for specific outputs

**Status**: Not implemented. Priority: Medium.

### Gap 6: Error Messages

When a typed action fails at build time, the error points to bash:

```
error: builder for '/nix/store/...-foo.drv' failed
  > patchelf: cannot find ELF header
  > error: command 'patchelf' failed with exit code 1
```

**Needed**: Map back to Haskell source location:

```
error: builder for '/nix/store/...-foo.drv' failed
  > at PatchElf.setRpath (foo.hs:47)
  > patchelf: cannot find ELF header
```

Implementation:

1. Add source locations to Action type (via Template Haskell or HasCallStack)
1. Serialize locations in actionToNix
1. Include in error output from actionToShell

**Status**: Not designed. Priority: Low (nice to have).

### Migration Strategy

**Phase A: Incremental Tool Adoption (Now → 4 weeks)**

1. Implement `aleph.phases.interpret` bridge
1. Add `Aleph.Nix.Tools.Wrap` module
1. Add `Chmod` action
1. Port one complex package (e.g., cudnn) using mixed approach
1. Document migration patterns

**Phase B: Full Package Migration (4 → 12 weeks)**

1. Port NVIDIA SDK packages to pure .hs
1. Add CMake/Meson typed options
1. Implement cross-compilation support
1. Implement multiple outputs
1. Port C++ packages (fmt, spdlog, etc.)

**Phase C: Default Everywhere (12+ weeks)**

1. Add lint rule: warn on bash phases > 3 lines
1. Add lint rule: error on heredocs in phases
1. Document all typed tools
1. PureScript backend for frontend developers
1. Deprecate `writeShellApplication` in new code

### Success Criteria

**Typed packages are "default everywhere" when:**

1. ✅ `call-package ./foo.hs {}` works
1. ✅ Tool deps are automatically tracked
1. ⬜ Existing packages can incrementally adopt typed phases
1. ⬜ All common tools have typed modules
1. ⬜ Cross-compilation works
1. ⬜ Multiple outputs work
1. ⬜ Zero bash phases in new packages (by convention)
1. ⬜ Lint rules enforce typed-first approach

## References

- [ℵ-003: The Straylight Prelude](aleph-003-prelude.md)
- [ℵ-007: Nix Formalization](aleph-007-formalization.md)
- [Shelly documentation](https://hackage.haskell.org/package/shelly)
- [Dhall language](https://dhall-lang.org/)
- [turtle: Shell programming in Haskell](https://hackage.haskell.org/package/turtle)
- [GHC WASM Backend](https://ghc.gitlab.haskell.org/ghc/doc/users_guide/wasm.html)
