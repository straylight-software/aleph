# Standard Environments

Build environments with real debug info. The "Turing Registry" defines flags that make code debuggable.

## Turing Registry

Source: `nix/prelude/turing-registry.nix`

All aleph stdenvs include these flags:

```bash
# Optimization
-O2                           # Real optimizations, debugger can follow

# Debug info
-g3                           # Maximum info (includes macros)
-gdwarf-5                     # Modern DWARF format
-fno-limit-debug-info         # Don't truncate for speed
-fstandalone-debug            # Full info for system headers

# Frame pointers (stack traces work)
-fno-omit-frame-pointer       # Keep rbp/x29
-mno-omit-leaf-frame-pointer  # Even in leaf functions

# No security theater
-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0
-fno-stack-protector
-fno-stack-clash-protection
-fcf-protection=none          # x86 only
```

Derivation attributes:

```nix
{
  __structuredAttrs = true;   # JSON instead of env vars
  dontStrip = true;           # Keep debug symbols
  separateDebugInfo = false;  # Symbols stay in binary
  hardeningDisable = ["all"]; # No hardening flags
  noAuditTmpdir = true;       # Don't audit /tmp references
}
```

## Available Stdenvs

Access via `config.aleph.prelude.stdenv` or `pkgs.aleph.stdenv`:

### Linux

| Name | Compiler | Libc | Linking | Notes |
|------|----------|------|---------|-------|
| `clang-glibc-dynamic` | Clang | glibc | Dynamic | Default on Linux |
| `clang-glibc-static` | Clang | glibc | Static | |
| `clang-musl-dynamic` | Clang | musl | Dynamic | |
| `clang-musl-static` | Clang | musl | Static | Portable binaries |
| `gcc-glibc-dynamic` | GCC 15/14/13 | glibc | Dynamic | |
| `gcc-glibc-static` | GCC | glibc | Static | |
| `gcc-musl-dynamic` | GCC | musl | Dynamic | |
| `gcc-musl-static` | GCC | musl | Static | |
| `nvidia` | Clang+CUDA | glibc | Dynamic | GPU builds |

Aliases:

- `default` = `clang-glibc-dynamic`
- `static` = `clang-glibc-static`
- `portable` = `clang-musl-static`

### Darwin

| Name | Compiler | Notes |
|------|----------|-------|
| `darwin` | Apple Clang | Default on Darwin |

## Usage

```nix
perSystem = { config, ... }:
  let
    inherit (config.aleph.prelude) stdenv;
  in {
    packages.my-tool = stdenv.default {
      pname = "my-tool";
      version = "0.1.0";
      src = ./.;
    };

    packages.portable = stdenv.portable {
      pname = "my-tool";
      version = "0.1.0";
      src = ./.;
    };
  };
```

## Cross-Compilation

Source: `nix/prelude/cross.nix`

Access via `config.aleph.prelude.cross` or `pkgs.aleph.cross`:

### From x86_64

| Target | Architecture | GPU |
|--------|--------------|-----|
| `grace` | aarch64 | Hopper (sm_90a) |
| `jetson` | aarch64 | Thor (sm_90) |
| `aarch64` | aarch64 | None |

### From aarch64

| Target | Architecture | GPU |
|--------|--------------|-----|
| `x86-64` | x86_64 | Blackwell (sm_120) |

Usage:

```nix
perSystem = { config, ... }:
  let
    inherit (config.aleph.prelude) cross;
  in {
    packages.grace-build = cross.grace {
      pname = "my-tool";
      version = "0.1.0";
      src = ./.;
    };
  };
```

## GPU Architectures

Source: `nix/prelude/gpu.nix`

| Arch | Capability | Name | Generation |
|------|------------|------|------------|
| `sm_120` | 12.0 | Blackwell | 12 |
| `sm_90a` | 9.0 | Hopper | 9 |
| `sm_90` | 9.0 | Thor | 9 |
| `sm_89` | 8.9 | Ada | 8 |
| `sm_86` | 8.6 | Ampere | 8 |

Capabilities:

- `supports-fp8`: generation >= 9
- `supports-nvfp4`: generation >= 12
- `supports-tma`: generation >= 9

## Platform Detection

Source: `nix/prelude/platform.nix`

```nix
pkgs.aleph.platform.current      # Current platform info
pkgs.aleph.platform.is-linux     # true on Linux
pkgs.aleph.platform.is-darwin    # true on Darwin
pkgs.aleph.platform.is-x86       # true on x86_64
pkgs.aleph.platform.is-arm       # true on aarch64
```
