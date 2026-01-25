# Aleph-Naught (aleph)

A standard library for Nix. Functional primitives, typed shell scripts, and build environments for software that needs to work.

## What It Provides

**Prelude** - Haskell-style functions for Nix: `map`, `filter`, `fold`, `maybe`, `either`. Consistent naming (`lisp-case`), predictable argument order (data last).

**Stdenvs** - Build environments with real debug info. Clang/GCC, glibc/musl, static/dynamic. NVIDIA CUDA when available. All with `-O2 -g3 -gdwarf-5`.

**Aleph.Script** - Haskell scripts instead of bash. `Aleph.Script` wraps Shelly with Turtle ergonomics. 32 scripts for containers, VMs, GPU passthrough.

**Overlays** - Composable package modifications: `prelude`, `container`, `script`, `libmodern`, `ghc-wasm`, `llvm-git`, `nvidia-sdk`.

**Flake Modules** - Drop-in flake-parts modules for formatter, linter, devshell, docs, NVIDIA SDK.

## Quick Start

```nix
{
  inputs.aleph.url = "github:straylight-software/aleph";

  outputs = inputs@{ flake-parts, aleph, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ aleph.modules.flake.default ];
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];

      perSystem = { config, pkgs, ... }:
        let inherit (config.aleph) prelude; in
        {
          packages.default = prelude.stdenv.default {
            pname = "example";
            version = "0.1.0";
            src = ./.;
          };
        };
    };
}
```

## Documentation Structure

- [**Start**](start/README.md) - Installation and first flake
- [**Concepts**](concepts/README.md) - Prelude, overlays, stdenvs, typed unix
- [**Guides**](guides/README.md) - How to write packages, modules, scripts
- [**Reference**](reference/README.md) - Modules, scripts, templates
- [**RFCs**](rfc/README.md) - Design decisions and rationale
