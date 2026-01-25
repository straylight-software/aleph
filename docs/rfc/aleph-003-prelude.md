# ℵ-003: The Straylight Prelude

| Field | Value |
|-------|-------|
| RFC | ℵ-003 |
| Title | The Straylight Prelude |
| Author | Straylight |
| Status | Draft |
| Created | 2025-01-13 |

## Abstract

This RFC specifies the **Straylight Prelude**, the default environment for all Straylight Standard Nix
code. The prelude provides a minimal, orthogonal, memorable interface that:

1. Eliminates camelCase from user-facing code
1. Pins language versions to known-good configurations
1. Exports toolchain bundles consumable by Nix, Bazel, and Buck2
1. Enforces structured attrs and content-addressed derivations by default

## Motivation

The Nix ecosystem suffers from:

- Inconsistent naming (camelCase, snake_case, lisp-case mixed)
- Version chaos (python3, python311, python312, python3Packages)
- Configuration sprawl (every flake configures nixpkgs differently)
- String-based derivations (no structure, no content-addressing)

The prelude is a **membrane** between straylight code and the nixpkgs substrate. Users write
lisp-case, structured, version-pinned code. The membrane translates to what Nix expects.

## Specification

### 1. Platform

The target execution environment.

```nix
platform.linux-x86-64
platform.linux-sbsa          # ARM64 datacenter (Grace)
platform.linux-aarch64       # ARM64 consumer/embedded
platform.darwin-aarch64      # Apple Silicon
platform.darwin-x86-64       # Intel Mac (legacy)
```

### 2. Stdenv

The build environment. A triple of `compiler × libc × linkage`.

```nix
stdenv.clang-glibc-dynamic   # CUDA ok
stdenv.clang-glibc-static    # CUDA ok
stdenv.clang-musl-dynamic    # no CUDA
stdenv.clang-musl-static     # no CUDA, portable

stdenv.gcc-glibc-dynamic     # CUDA ok
stdenv.gcc-glibc-static      # CUDA ok
stdenv.gcc-musl-dynamic      # no CUDA
stdenv.gcc-musl-static       # no CUDA, portable
```

#### 2.1 The Law (Not Configurable)

Every stdenv includes, non-negotiably:

```
# Optimization
-O2

# Debug symbols
-g3 -gdwarf-5 -fno-limit-debug-info -fstandalone-debug

# Frame pointers (stack traces work)
-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer

# No security theater
-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0
-fno-stack-protector -fno-stack-clash-protection
-fcf-protection=none  # x86 only

# Nix attrs
dontStrip = true
separateDebugInfo = false
hardeningDisable = [ "all" ]

# C++ standard
-std=c++23
```

#### 2.2 Aliases

```nix
stdenv.default  = stdenv.clang-glibc-dynamic
stdenv.static   = stdenv.clang-glibc-static
stdenv.portable = stdenv.clang-musl-static
```

#### 2.3 Constraint

```
gpu != none → libc == glibc
```

CUDA requires glibc. Musl is for portable CPU-only binaries.

### 3. GPU

Target GPU architecture.

```nix
gpu.sm_120       # Blackwell
gpu.sm_90a       # Hopper
gpu.sm_89        # Ada
gpu.sm_90        # Thor
gpu.none         # CPU only
```

### 4. Languages

Pinned versions. One version. The right version.

```nix
python           # 3.12 (NVIDIA lower bound)
ghc              # 9.10
lean             # 4.15.0
rust             # 1.92.0
clang            # 20 (LLVM from git, SM120 Blackwell support)
gcc              # 15 (libstdc++ for C++23)
```

#### 4.1 Rationale

| Language | Version | Reason |
|----------|---------|--------|
| Python | 3.12 | NVIDIA SDK lower bound |
| GHC | 9.10 | Latest stable with good libraries |
| Lean | 4.15.0 | Mathlib compatible |
| Rust | 1.92.0 | Latest stable |
| Clang | 20 | LLVM from git for SM120 Blackwell support |
| GCC | 15 | libstdc++ with C++23 for device code |

#### 4.2 Language Namespaces

Each language has one way to build. Inputs and outputs are named differently.

```nix
python = {
  build              # produce a python package
  version            # "3.12"
  
  # inputs (what you depend on)
  pkgs               # python.pkgs.numpy, python.pkgs.torch
  
  # outputs (what you produce)
  package            # importable package
  app                # executable application
  lib                # C extension
  
  # compositions
  cuda.build         # with CUDA support
  cuda.pkgs          # torch, cupy with CUDA
  native.build       # with C/C++ extensions
};

ghc = {
  build
  version            # "9.10"
  
  pkgs               # ghc.pkgs.aeson, ghc.pkgs.mtl
  
  package
  app
  lib
  
  ffi.build          # with c2hs/hsc2hs
};

rust = {
  build
  version            # "1.92.0"
  
  crates             # rust.crates.serde, rust.crates.tokio
  
  package
  bin
  lib
  staticlib          # for C FFI
  
  pyo3.build         # Python bindings via maturin
};

lean = {
  build
  version            # "4.15.0"
  
  pkgs               # lean.pkgs.mathlib, lean.pkgs.std
  
  package
  lib
};

cpp = {
  # uses stdenv directly
  bin                # executable
  lib                # shared library
  staticlib          # static library
  header-only        # header-only library
  
  cuda.build         # CUDA device + host
  cuda.kernel        # device code only
  cuda.host          # host code only
};
```

#### 4.3 Usage

```nix
{ python, rust, cpp, gpu }:

python.cuda.build {
  pname = "straylight-inference";
  deps = [ python.cuda.pkgs.torch ];
}

rust.pyo3.build {
  pname = "straylight-fast";
}

cpp.cuda.kernel {
  pname = "attention-kernel";
  target-gpu = gpu.sm_120;
}
```

No version suffixes. No `python3`, `python312`. Just `python`.

#### 4.4 Pipe Operator

Nix now has pipe operators. Use them. Read left-to-right like a human.

```nix
# The old way (inside-out, backwards, painful)
lib.filterAttrs (n: v: v.meta.broken or false) 
  (lib.mapAttrs (n: v: v // { priority = 10; }) 
    (lib.attrByPath ["packages" system] {} flake))

# The new way (left-to-right, obvious, correct)
flake
|> lib.attrByPath ["packages" system] {}
|> map-attrs (n: v: v // { priority = 10; })
|> filter-attrs (n: v: v.meta.broken or false)
```

##### 4.4.1 Pipes with Prelude

The prelude's functional core is designed for pipes:

```nix
# Process a list
packages
|> filter (p: p.meta.license == license.mit)
|> map (p: p.name)
|> sort lt
|> join ", "

# Transform attrs
config.services
|> filter-attrs (n: v: v.enable)
|> map-attrs (n: v: v.package)
|> values
|> unique

# Build a package set
sources
|> map-attrs (n: src: stdenv.default { pname = n; inherit src; })
|> filter-attrs (n: v: v.meta.broken or false |> not)

# Chain fetches and transforms
fetch.github { owner = "straylight-software"; repo = "sdk"; rev = "v1.0.0"; hash = "..."; }
|> python.build { pname = "straylight-sdk"; deps = [ python.pkgs.torch ]; }
|> container.oci { name = "straylight-sdk"; }
```

##### 4.4.2 Backward Pipe

For when you have the argument first:

```nix
# Forward pipe: f |> g |> h  means  h(g(f))
# Backward pipe: h <| g <| f  means  h(g(f))

# Useful for:
assert (x |> validate) <| "validation failed: ${x}";

# Or inline arguments:
map (add 1) <| [1 2 3 4 5]
```

##### 4.4.3 A Better Life

Before (the dark times):

```nix
let
  pkgs' = import nixpkgs { inherit system; };
  filtered = lib.filterAttrs (n: v: lib.hasPrefix "straylight-" n) pkgs';
  mapped = lib.mapAttrs (n: v: v.overrideAttrs (old: { 
    meta = old.meta // { priority = 10; }; 
  })) filtered;
  list = lib.mapAttrsToList (n: v: { name = n; value = v; }) mapped;
  sorted = lib.sort (a: b: a.name < b.name) list;
in
lib.listToAttrs sorted
```

After (enlightenment):

```nix
pkgs
|> filter-attrs (n: v: n |> starts-with "straylight-")
|> map-attrs (n: v: v.override-attrs (old: { 
     meta = old.meta // { priority = 10; }; 
   }))
|> to-list
|> sort (a: b: a.name |> lt b.name)
|> from-list
```

The operations flow. The data transforms. You read it like prose.

### 5. Build Systems

#### 5.1 Acceptable

```nix
make             # honest
just             # honest, modern
setup-py         # python's way
cargo            # rust's way
cabal            # haskell's way
lake             # lean's way
meson            # acceptable
```

#### 5.2 Banned

```nix
# cmake          # BANNED
```

**THE GRIMY CMAKE ENJOYERS IS OUTLAWED.**

CMake is a liar. It hides dependency information in inscrutable generator expressions,
creates platform-specific build trees, and makes reproducibility nearly impossible.

We don't negotiate with CMake. We extract its confessions.

#### 5.3 The Confessor

For upstreams that use CMake, we extract the truth and discard the sinner:

```nix
cmake-to-pc      # cmake project → pkg-config files
```

The `cmake-to-pc` tool:

1. Runs CMake configure to extract target information
1. Generates proper `.pc` files for each library
1. Discards the CMake build tree
1. Builds with make/ninja using the extracted information

```nix
{ cmake-to-pc, stdenv }:

stdenv.default {
  pname = "libfoo";
  
  # Extract pkg-config from CMake, then build honestly
  native-deps = [ cmake-to-pc ];
  
  configure-phase = ''
    cmake-to-pc extract . --output pc/
  '';
  
  build-phase = ''
    # Now we have honest .pc files
    make -j$NIX_BUILD_CORES
  '';
}
```

Upstreams using CMake are not rejected. Their sins are confessed, their lies extracted
into truth, and then they are built correctly.

### 6. Fetch

Retrieve sources. No heredocs.

```nix
fetch.github     # { owner, repo, rev, hash }
fetch.gitlab     # { owner, repo, rev, hash }
fetch.git        # { url, rev, hash }
fetch.url        # { url, hash }
fetch.tarball    # { url, hash }
fetch.fod        # { name, hash, script } — escape hatch
```

### 7. Render

Serialize data to files. No heredocs.

```nix
render.json      # attrs → /nix/store/...-name.json
render.toml      # attrs → /nix/store/...-name.toml
render.yaml      # attrs → /nix/store/...-name.yaml
render.ini       # attrs → /nix/store/...-name.ini
render.env       # attrs → /nix/store/...-name.env
render.nix-conf  # attrs → /nix/store/...-nix.conf
render.systemd   # attrs → systemd unit
render.nginx     # attrs → nginx config
```

### 8. Script

Wrap foreign code. No heredocs.

```nix
script.bash      # { name, deps, src }
script.python    # { name, deps, src }
script.c         # { name, src }
```

All scripts read from files, not inline strings.

### 9. Opt

Module options in lisp-case.

```nix
opt.enable       # lib.mkEnableOption
opt.str          # lib.mkOption { type = lib.types.str; }
opt.int          # lib.mkOption { type = lib.types.int; }
opt.bool         # lib.mkOption { type = lib.types.bool; }
opt.path         # lib.mkOption { type = lib.types.path; }
opt.port         # lib.mkOption { type = lib.types.port; }
opt.list-of      # lib.mkOption { type = lib.types.listOf ...; }
opt.attrs-of     # lib.mkOption { type = lib.types.attrsOf ...; }
opt.one-of       # lib.mkOption { type = lib.types.enum ...; }
opt.package      # lib.mkOption { type = lib.types.package; }
```

### 10. When

Conditionals in lisp-case.

```nix
when             # lib.mkIf
when-attr        # lib.optionalAttrs
when-list        # lib.optional / lib.optionals
when-str         # lib.optionalString
```

### 11. Functional Core

The prelude exposes a sane functional vocabulary. If you know Haskell, you know this.

#### 11.1 Fundamentals

```nix
id               # a → a
const            # a → b → a
flip             # (a → b → c) → b → a → c
compose          # (b → c) → (a → b) → a → c  (also: ∘)
pipe             # (a → b) → (b → c) → a → c  (flip compose)
```

#### 11.2 Lists

```nix
map              # (a → b) → [a] → [b]
filter           # (a → bool) → [a] → [a]
fold             # (b → a → b) → b → [a] → b
fold-right       # (a → b → b) → b → [a] → b
head             # [a] → a
tail             # [a] → [a]
init             # [a] → [a]
last             # [a] → a
take             # int → [a] → [a]
drop             # int → [a] → [a]
length           # [a] → int
reverse          # [a] → [a]
concat           # [[a]] → [a]
flatten          # nested → [a]
concat-map       # (a → [b]) → [a] → [b]
zip              # [a] → [b] → [(a, b)]
zip-with         # (a → b → c) → [a] → [b] → [c]
sort             # (a → a → bool) → [a] → [a]
unique           # [a] → [a]
elem             # a → [a] → bool
find             # (a → bool) → [a] → a | null
partition        # (a → bool) → [a] → { right : [a]; wrong : [a]; }
group-by         # (a → string) → [a] → attrs
range            # int → int → [int]
replicate        # int → a → [a]
```

#### 11.3 Attrs

```nix
map-attrs        # (string → a → b) → attrs → attrs
filter-attrs     # (string → a → bool) → attrs → attrs
fold-attrs       # (b → string → a → b) → b → attrs → b
keys             # attrs → [string]
values           # attrs → [a]
has              # string → attrs → bool
get              # [string] → attrs → a → a  (with default)
get'             # string → attrs → a        (throws)
set              # [string] → a → attrs → attrs
remove           # [string] → attrs → attrs
merge            # attrs → attrs → attrs
merge-all        # [attrs] → attrs
to-list          # attrs → [{ name : string; value : a; }]
from-list        # [{ name : string; value : a; }] → attrs
map-to-list      # (string → a → b) → attrs → [b]
intersect        # attrs → attrs → attrs
gen-attrs        # [string] → (string → a) → attrs
```

#### 11.4 Strings

```nix
split            # string → string → [string]
join             # string → [string] → string
trim             # string → string
replace          # [string] → [string] → string → string
starts-with      # string → string → bool
ends-with        # string → string → bool
contains         # string → string → bool
to-lower         # string → string
to-upper         # string → string
to-string        # a → string
string-length    # string → int
substring        # int → int → string → string
```

#### 11.5 Maybe (null handling)

```nix
maybe            # b → (a → b) → a | null → b
from-maybe       # a → a | null → a
is-null          # a | null → bool
cat-maybes       # [a | null] → [a]
map-maybe        # (a → b | null) → [a] → [b]
```

#### 11.6 Comparison

```nix
eq               # a → a → bool
neq              # a → a → bool
lt               # a → a → bool
le               # a → a → bool
gt               # a → a → bool
ge               # a → a → bool
min              # a → a → a
max              # a → a → a
compare          # a → a → ordering  (-1, 0, 1)
clamp            # a → a → a → a    (lo hi x → bounded x)
```

#### 11.7 Boolean

```nix
not              # bool → bool
and              # bool → bool → bool
or               # bool → bool → bool
all              # (a → bool) → [a] → bool
any              # (a → bool) → [a] → bool
```

#### 11.8 Arithmetic

```nix
add              # int → int → int
sub              # int → int → int
mul              # int → int → int
div              # int → int → int
mod              # int → int → int
neg              # int → int
abs              # int → int
sum              # [int] → int
product          # [int] → int
```

### 12. License

Common licenses.

```nix
license.mit
license.asl-2-0
license.bsd-3
license.gpl-3
license.mpl-2-0
license.unfree
```

### 13. Attr Translation

The prelude translates lisp-case attrs to camelCase for Nix internals.

| User Writes | Nix Sees |
|-------------|----------|
| `build-inputs` | `buildInputs` |
| `native-build-inputs` | `nativeBuildInputs` |
| `propagated-build-inputs` | `propagatedBuildInputs` |
| `check-inputs` | `checkInputs` |
| `pre-configure` | `preConfigure` |
| `configure-flags` | `configureFlags` |
| `post-configure` | `postConfigure` |
| `pre-build` | `preBuild` |
| `build-flags` | `buildFlags` |
| `post-build` | `postBuild` |
| `pre-install` | `preInstall` |
| `install-flags` | `installFlags` |
| `post-install` | `postInstall` |
| `pre-check` | `preCheck` |
| `check-flags` | `checkFlags` |
| `post-check` | `postCheck` |
| `pre-fixup` | `preFixup` |
| `post-fixup` | `postFixup` |
| `make-flags` | `makeFlags` |
| `cmake-flags` | `cmakeFlags` |
| `meson-flags` | `mesonFlags` |
| `meta.main-program` | `meta.mainProgram` |

### 14. Nix Config

The prelude exports nix configuration for `nix.conf`:

```nix
nix-config = {
  experimental-features = [
    "nix-command"
    "flakes" 
    "ca-derivations"
  ];
  content-addressed-by-default = true;
};
```

### 15. Structured Attrs and CA Derivations

Every `mkDerivation` in the prelude:

- Sets `__structuredAttrs = true`
- ~~Sets `__contentAddressed = true`~~ (temporarily disabled pending nix fork)
- ~~Uses `outputHashMode = "recursive"`~~ (temporarily disabled pending nix fork)

Users never see these. They're axioms.

> **Status**: CA derivations are temporarily disabled until the aleph nix fork
> lands with improved CA support. Structured attrs remain enabled.

### 16. Bundle Export

The prelude exports a bundle for consumption by Bazel and Buck2:

```nix
bundle = {
  nix-config
  platform
  stdenv
  gpu
  
  versions = {
    python = "3.12";
    ghc = "9.10";
    lean = "4.15.0";
    rust = "1.92.0";
    clang = "20";
    gcc = "15";
  };
  
  paths = {
    cc.compiler
    cc.linker
    cc.ar
    cc.stdlib
    cc.headers
    
    cuda.nvcc
    cuda.runtime
    cuda.headers
    cuda.libdevice
    
    python.interpreter
    python.stdlib
    
    rust.rustc
    rust.cargo
    rust.stdlib
    
    haskell.ghc
    haskell.cabal
    
    lean.lean
    lean.lake
    
    shell.bash
    shell.coreutils
  };
};
```

Starlark reads `render.json bundle` and configures toolchains accordingly.

## Example

A complete package using only the prelude:

```nix
{ stdenv, fetch, render, python, license, platform, when-list }:

stdenv.default {
  pname = "straylight-tool";
  version = "1.0.0";
  
  src = fetch.github {
    owner = "straylight-software";
    repo = "straylight-tool";
    rev = "v1.0.0";
    hash = "sha256-...";
  };
  
  native-build-inputs = [ cmake ninja ];
  build-inputs = [ openssl zlib ] ++ when-list (platform == platform.linux-x86-64) [ numactl ];
  
  cmake-flags = [
    "-DPYTHON_EXECUTABLE=${python}/bin/python"
  ];
  
  post-install = ''
    cp ${render.json "config.json" { 
      version = "1.0.0";
      features = [ "cuda" "distributed" ];
    }} $out/etc/
  '';
  
  meta = {
    description = "Straylight inference tool";
    license = license.mit;
    main-program = "straylight-tool";
  };
}
```

No camelCase. No version suffixes. No heredocs. No configuration. Just code.

## Conformance

Code is **Straylight Prelude conformant** if it:

1. Uses only prelude-provided bindings for stdenv, languages, and utilities
1. Contains no camelCase in user-written code
1. Contains no heredocs
1. Contains no inline scripts over 10 lines
1. Does not override pinned language versions
1. Does not disable structured attrs or CA derivations

## Coset Structure

The prelude is the **kernel** — the intersection that works across all supports:

```
Build Systems: { nixos, buck2, bazel }
Languages:     { c++, cuda, python, rust, haskell, lean4, bash }
```

Each build system support extends the prelude:

- **NixOS**: Uses stdenv directly, adds systemd/networking/filesystems
- **Bazel**: Maps bundle to `@local_config_*` overrides via Starlark
- **Buck2**: Maps bundle to Buck2 toolchain configs via Starlark

The invariant: **same bundle → same binaries** regardless of which build system consumes it.

## References

- [ℵ-001: Straylight Standard Nix](aleph-001-standard-nix.md)
- [ℵ-002: Linting](aleph-002-lint.md)
