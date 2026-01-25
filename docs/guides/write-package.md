# Write a Package

Build a derivation using `aleph` `stdenv`s.

## Basic Package

```nix
perSystem = { config, pkgs, ... }:
  let
    inherit (config.aleph.prelude) stdenv;
  in {
    packages.my-tool = stdenv.default {
      pname = "my-tool";
      version = "0.1.0";
      src = ./.;

      native-build-inputs = [ pkgs.cmake ];
      build-inputs = [ pkgs.zlib ];

      meta = {
        description = "My tool";
        license = pkgs.lib.licenses.mit;
      };
    };
  };
```

## Choosing a Stdenv

```nix
# Default (clang, glibc, dynamic)
stdenv.default { ... }

# Static linking (clang, glibc)
stdenv.static { ... }

# Portable binary (clang, musl, static)
stdenv.portable { ... }

# GCC instead of Clang
stdenv.gcc-glibc-dynamic { ... }

# NVIDIA with C++23 device code and `std::mdspan`.
stdenv.nvidia { ... }
```

## Cross-Compilation

```nix
let
  inherit (config.aleph.prelude) cross;
in {

  # Build for Grace Hopper (aarch64 + Hopper GPU)
  packages.grace-build = cross.grace {
    pname = "my-tool";
    version = "0.1.0";
    src = ./.;
  };

  # Build for generic aarch64
  packages.arm64-build = cross.aarch64 {
    pname = "my-tool";
    version = "0.1.0";
    src = ./.;
  };
}
```

## Raw `stdenv` Access

Skip `aleph` wrappers and use `nixpkgs` directly:

```nix
stdenv.default.raw {
  name = "example";
  # ... standard mkDerivation args
}
```
