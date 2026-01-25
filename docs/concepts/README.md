# Concepts

Core ideas behind aleph.

## [Prelude](prelude.md)

A functional library for Nix. 100+ functions with Haskell-style naming and semantics.

- **Fundamentals**: `id`, `const`, `flip`, `compose`, `pipe`, `fix`
- **Lists**: `map`, `filter`, `fold`, `head`, `tail`, `take`, `drop`, `zip`, `sort`, `unique`
- **Attrs**: `map-attrs`, `filter-attrs`, `keys`, `values`, `get`, `set`, `merge`, `to-list`
- **Strings**: `split`, `join`, `trim`, `replace`, `starts-with`, `ends-with`, `to-lower`
- **Maybe**: `maybe`, `from-maybe`, `is-null`, `cat-maybes`, `map-maybe`
- **Either**: `left`, `right`, `is-left`, `is-right`, `either`, `from-right`
- **Types**: `is-list`, `is-attrs`, `is-string`, `is-int`, `is-function`, `typeof`

## [Overlays](overlays.md)

Composable nixpkgs transformations. Eight overlays that can be used individually or together.

- **prelude**: `pkgs.aleph.prelude`, `pkgs.aleph.stdenv`, `pkgs.aleph.cross`
- **container**: `pkgs.aleph.container.mk-namespace-env`, `unshare-run`, `fhs-run`, `gpu-run`
- **script**: `pkgs.aleph.script.ghc`, `gen-wrapper`, `check`, compiled scripts
- **libmodern**: `pkgs.libmodern.fmt`, `pkgs.libmodern.abseil-cpp`, `pkgs.libmodern.libsodium`
- **ghc-wasm**: GHC WASM backend for `builtins.wasm` integration
- **llvm-git**: LLVM from git with SM120 Blackwell support
- **nvidia-sdk**: CUDA 13.x packages
- **packages**: `pkgs.mdspan`

## [Stdenvs](stdenvs.md)

Build environments with real debug info. The "Turing Registry" - flags that make code debuggable.

| Name | Compiler | Libc | Linking |
|------|----------|------|---------|
| `clang-glibc-dynamic` | Clang | glibc | Dynamic |
| `clang-glibc-static` | Clang | glibc | Static |
| `clang-musl-dynamic` | Clang | musl | Dynamic |
| `clang-musl-static` | Clang | musl | Static |
| `gcc-glibc-dynamic` | GCC | glibc | Dynamic |
| `gcc-musl-static` | GCC | musl | Static |
| `nvidia` | Clang+CUDA | glibc | Dynamic |

All stdenvs include: `-O2 -g3 -gdwarf-5 -fno-omit-frame-pointer`

## [Aleph.Script](typed-unix.md)

Haskell scripts instead of bash. `Aleph.Script` combines Shelly's foundation with Turtle's ergonomics.

```haskell
import Aleph.Script

main :: IO ()
main = script $ do
    files <- ls "."
    for_ files $ \f ->
        when (hasExtension "nix" f) $
            echo $ format ("Found: "%fp) f
```

32 compiled scripts for containers, VMs, GPU passthrough, linting.
