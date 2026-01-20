# Overlays

Composable nixpkgs transformations. Each overlay adds functionality to the package set.

## Available Overlays

### prelude

Source: `nix/prelude/default.nix`

Adds the straylight namespace to pkgs:

```nix
pkgs.straylight.prelude      # Functional library
pkgs.straylight.platform     # Platform detection
pkgs.straylight.gpu          # GPU architecture metadata
pkgs.straylight.turing-registry  # Build flags
pkgs.straylight.stdenv       # Build environment matrix
pkgs.straylight.cross        # Cross-compilation targets
pkgs.straylight.toolchain    # Compiler paths
pkgs.straylight.info         # Introspection
```

### container

Source: `nix/overlays/container/default.nix`

Container and namespace utilities:

```nix
pkgs.straylight.container.mk-namespace-env   # Create namespace runners
pkgs.straylight.container.mk-oci-rootfs      # Extract OCI images
pkgs.straylight.container.mk-firecracker-image  # Build VM disk images
pkgs.straylight.container.mk-simple-index    # Generate PEP 503 indexes
pkgs.straylight.container.extract            # Binary extraction with patchelf
pkgs.straylight.container.oci-run            # Run OCI images in namespaces
pkgs.straylight.container.fhs-run            # Run with FHS layout
pkgs.straylight.container.gpu-run            # Run with GPU access
```

### script

Source: `nix/overlays/script.nix`

Haskell scripting infrastructure:

```nix
pkgs.straylight.script.ghc           # GHC with Aleph.Script
pkgs.straylight.script.gen-wrapper   # Generate typed CLI wrappers
pkgs.straylight.script.check         # Validation script
pkgs.straylight.script.props         # Property tests
pkgs.straylight.script.shell         # Development shell
pkgs.straylight.script.compiled.*    # Pre-compiled scripts
pkgs.straylight.script.nix-dev       # Development Nix wrapper
pkgs.straylight.script.nix-ci        # CI Nix wrapper
```

Compiled scripts:

- `vfio-bind`, `vfio-unbind`, `vfio-list` - GPU passthrough
- `oci-run`, `oci-gpu`, `oci-inspect`, `oci-pull` - Container operations
- `fhs-run`, `gpu-run` - Namespace runners
- `fc-run`, `fc-build` - Firecracker VMs
- `ch-run`, `ch-gpu` - Cloud Hypervisor VMs

### libmodern

Source: `nix/overlays/libmodern/default.nix`

Modern C++ libraries built with straylight stdenvs:

```nix
pkgs.libmodern.fmt           # fmt 11.1.4
pkgs.libmodern.libsodium     # libsodium 1.0.20
pkgs.libmodern.abseil-cpp    # abseil 20250127.1 (combined static archive)
```

### ghc-wasm

Source: `nix/overlays/ghc-wasm.nix`

GHC WASM backend for `builtins.wasm` integration.

### llvm-git

Source: `nix/overlays/llvm-git.nix`

LLVM from git with SM120 Blackwell GPU support.

### nvidia-sdk

Source: `nix/overlays/nixpkgs-nvidia-sdk.nix`

CUDA 13.x packages.

### packages

Source: `nix/overlays/packages/`

Additional packages:

```nix
pkgs.mdspan    # C++23 mdspan header-only library
```

## Composition

The default overlay composes all overlays:

```nix
flake.overlays.default = lib.composeManyExtensions [
  packages-overlay
  llvm-git-overlay
  nvidia-sdk-overlay
  prelude-overlay
  container-overlay
  script-overlay
  ghc-wasm-overlay
  libmodern-overlay
];
```

## Using Individual Overlays

```nix
{
  nixpkgs.overlays = [
    aleph.overlays.prelude
    aleph.overlays.script
  ];
}
```

## Using the Default Overlay

```nix
{
  nixpkgs.overlays = [ aleph.overlays.default ];
}
```

Or via the flake module (recommended):

```nix
{
  imports = [ aleph.modules.flake.default ];
}
```
