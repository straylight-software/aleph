# Overlays

Composable nixpkgs transformations. Each overlay adds functionality to the package set.

## Available Overlays

### prelude

Source: `nix/prelude/default.nix`

Adds the aleph namespace to pkgs:

```nix
pkgs.aleph.prelude      # Functional library
pkgs.aleph.platform     # Platform detection
pkgs.aleph.gpu          # GPU architecture metadata
pkgs.aleph.turing-registry  # Build flags
pkgs.aleph.stdenv       # Build environment matrix
pkgs.aleph.cross        # Cross-compilation targets
pkgs.aleph.toolchain    # Compiler paths
pkgs.aleph.info         # Introspection
```

### container

Source: `nix/overlays/container/default.nix`

Container and namespace utilities:

```nix
pkgs.aleph.container.mk-namespace-env   # Create namespace runners
pkgs.aleph.container.mk-oci-rootfs      # Extract OCI images
pkgs.aleph.container.mk-firecracker-image  # Build VM disk images
pkgs.aleph.container.mk-simple-index    # Generate PEP 503 indexes
pkgs.aleph.container.extract            # Binary extraction with patchelf
pkgs.aleph.container.unshare-run        # Run OCI images in bwrap/unshare namespaces
pkgs.aleph.container.fhs-run            # Run with FHS layout
pkgs.aleph.container.gpu-run            # Run with GPU access
```

### script

Source: `nix/overlays/script.nix`

Haskell scripting infrastructure:

```nix
pkgs.aleph.script.ghc           # GHC with Aleph.Script
pkgs.aleph.script.gen-wrapper   # Generate typed CLI wrappers
pkgs.aleph.script.check         # Validation script
pkgs.aleph.script.props         # Property tests
pkgs.aleph.script.shell         # Development shell
pkgs.aleph.script.compiled.*    # Pre-compiled scripts
pkgs.aleph.script.nix-dev       # Development Nix wrapper
pkgs.aleph.script.nix-ci        # CI Nix wrapper
```

Compiled scripts:

- `vfio-bind`, `vfio-unbind`, `vfio-list` - GPU passthrough
- `unshare-run`, `unshare-gpu`, `crane-inspect`, `crane-pull` - Container operations
- `fhs-run`, `gpu-run` - Namespace runners
- `isospin-run`, `isospin-build` - Firecracker VMs
- `cloud-hypervisor-run`, `cloud-hypervisor-gpu` - Cloud Hypervisor VMs

### libmodern

Source: `nix/overlays/libmodern/default.nix`

Modern C++ libraries built with aleph stdenvs:

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
