# aleph-naught Flake Review

**Date:** 2026-01-13
**Reviewer:** Claude (Anthropic)
**Status:** All issues resolved

______________________________________________________________________

## Executive Summary

aleph-naught is a Nix standard library providing:

- A functional prelude with Haskell-style naming (lisp-case)
- Stdenv matrix (clang/gcc × glibc/musl × dynamic/static)
- Cross-compilation targets (Grace Hopper, Jetson, aarch64, x86-64)
- GPU architecture metadata and CUDA support
- "The Law" - non-negotiable build flags for debuggability
- Documentation infrastructure (mdBook + NDG)
- Formatter configuration via treefmt-nix

______________________________________________________________________

## Review History

### First Review (Completed)

All 11 issues from the first review were fixed:

1. ✅ Template option names (`allowUnfree` → `allow-unfree`)
1. ✅ Template GPU typo (`sm120` → `sm_120`)
1. ✅ `render.toml/yaml/ini` functions (now return derivation directly)
1. ✅ NDG theme configuration (uses `cfg.theme` not hardcoded)
1. ✅ Compiler version fallbacks (`gcc15 or gcc14 or gcc13 or gcc`)
1. ✅ Templates use `follows` directives
1. ✅ Darwin support (wrapped Linux-only stdenvs)
1. ✅ Deduplicated `optionsOnlyModule` to `nix/flake-modules/options-schema.nix`
1. ✅ CUDA capability comments corrected
1. ✅ Added `_class = "flake"` to prelude module
1. ✅ `mkForce`/`mkDefault` stubs have runtime warnings

### Second Adversarial Review (Completed)

All 9 issues from the second review were fixed:

1. ✅ **CRITICAL: C builds fail** - Fixed by separating `NIX_CFLAGS_COMPILE` (C-only flags)
   from `NIX_CXXFLAGS_COMPILE` (`-std=c++23` for C++ only)
1. ✅ **CRITICAL: `translate-attrs` not applied** - Fixed in `mk-stdenv.__functor`
1. ✅ **CRITICAL: Wrong function name** - Changed `straylight.translateAttrs` to `straylight.translate-attrs`
1. ✅ **Docs cp command** - Fixed empty directory handling with `/. ` suffix
1. ✅ **NDG fallback** - Now fails gracefully when NDG is not available
1. ✅ **README/docs option names** - Updated to use lisp-case
1. ✅ **README GCC version** - Updated from 14 to 15
1. ✅ **Smoke tests** - Added C, C++, and mixed compilation smoke tests
1. ✅ **REVIEW.md** - Updated to reflect current state

______________________________________________________________________

## Smoke Tests Added

The following smoke tests verify the C/C++ flag separation:

- `checks.straylight-smoke-c` - Pure C compilation (must not receive `-std=c++23`)
- `checks.straylight-smoke-cpp` - C++23 compilation (must receive `-std=c++23`)
- `checks.straylight-smoke-mixed` - Mixed C/C++ compilation (each gets correct flags)

______________________________________________________________________

## Architecture Notes

### Strengths

1. **Clean separation of concerns**: Overlay provides primitives, flake-module provides ergonomics
1. **Lexical capture pattern**: Modules close over aleph-naught's inputs
1. **Comprehensive stdenv matrix**: Good coverage of compiler/libc/linkage combinations
1. **The Law**: Enforced debuggability is a strong opinion well-executed
1. **Lisp-case prelude**: Consistent naming convention
1. **Comprehensive test suite**: ~2800 lines of functional tests in demo-suite.nix

### File Structure

```
aleph-naught/
├── flake.nix                          # Entry point
├── nix/
│   ├── flake-modules/
│   │   ├── default.nix                # Plain Nix entry (for NDG)
│   │   ├── main.nix                   # All flake-parts modules
│   │   ├── options-schema.nix         # Shared options schema
│   │   └── nixpkgs-nvidia.nix         # NVIDIA SDK module
│   ├── lib/
│   │   ├── container.nix              # OCI/namespace utilities
│   │   └── default.nix                # Lib exports
│   ├── overlays/
│   │   ├── container.nix              # Container overlay
│   │   ├── default.nix                # Overlay composition
│   │   ├── prelude.nix                # The Weyl Prelude
│   │   └── nixpkgs-nvidia-sdk.nix     # NVIDIA SDK overlay
│   ├── prelude/
│   │   ├── default.nix                # Prelude package
│   │   ├── demo-suite.nix             # Test suite (~2800 lines)
│   │   ├── flake-module.nix           # Prelude as flake-parts module
│   │   ├── functions.nix              # Nixdoc-documented functions
│   │   └── lib.nix                    # nixpkgs/lib compatibility shim
│   └── templates/
│       ├── cuda/flake.nix
│       ├── default/flake.nix
│       └── minimal/flake.nix
├── docs/                              # mdBook documentation
└── docs-options/                      # NDG options documentation
```

______________________________________________________________________

## Commands

```bash
# Run all checks
nix flake check

# Build documentation
nix build .#docs

# Run specific smoke test
nix build .#checks.x86_64-linux.straylight-smoke-mixed

# Format
nix fmt
```
