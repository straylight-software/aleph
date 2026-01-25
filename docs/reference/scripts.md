# Compiled Scripts Reference

Source: `nix/scripts/*.hs`

32 Haskell scripts compiled for system operations.

## Container Operations

| Script | Dependencies | Description |
|--------|--------------|-------------|
| `unshare-run` | bubblewrap, crane, jq | Run OCI images in bwrap/unshare namespaces |
| `unshare-gpu` | bubblewrap, crane, jq, pciutils | Run OCI with GPU device access |
| `crane-inspect` | crane, jq | Inspect OCI image metadata |
| `crane-pull` | crane | Pull OCI images |

## Namespace Runners

| Script | Dependencies | Description |
|--------|--------------|-------------|
| `fhs-run` | bubblewrap | Run with FHS layout |
| `gpu-run` | bubblewrap, pciutils | Run with GPU access |

## VM Operations

| Script | Dependencies | Description |
|--------|--------------|-------------|
| `isospin-run` | firecracker | Run Firecracker VMs |
| `isospin-build` | e2fsprogs, cpio, gzip | Build Firecracker disk images |
| `cloud-hypervisor-run` | cloud-hypervisor | Run Cloud Hypervisor VMs |
| `cloud-hypervisor-gpu` | cloud-hypervisor, pciutils | Run with GPU passthrough |

## GPU Passthrough

| Script | Dependencies | Description |
|--------|--------------|-------------|
| `vfio-bind` | pciutils | Bind PCI devices to vfio-pci |
| `vfio-unbind` | pciutils | Unbind from vfio-pci |
| `vfio-list` | pciutils | List VFIO-capable devices |

## Development Tools

| Script | Dependencies | Description |
|--------|--------------|-------------|
| `check` | - | Validation script |
| `gen-wrapper` | - | Generate typed CLI wrappers |
| `gen-gnu-wrapper` | - | Generate GNU getopt wrappers |
| `gen-tool-wrapper` | - | Generate tool wrappers |
| `nix-dev` | nix | Development Nix wrapper |
| `nix-ci` | nix | CI Nix wrapper |
| `lint-init` | - | Initialize linting |
| `lint-link` | - | Link linting config |

## CLI Wrapper Tools

Supported tools for typed wrapper generation:

### Clap (Rust)

rg, fd, bat, delta, dust, tokei, hyperfine, deadnix, statix, stylua, taplo, zoxide

### GNU getopt_long

ls, grep, sed, find, xargs, tar, gzip, wget, rsync

### Hand-crafted

jq, crane, bwrap

## Usage

Via overlay:

```nix
pkgs.aleph.script.compiled.unshare-run
pkgs.aleph.script.compiled.fhs-run
pkgs.aleph.script.compiled.vfio-bind
```

Via devshell:

```nix
devShells.default = pkgs.mkShell {
  packages = with pkgs.aleph.script.compiled; [
    unshare-run
    fhs-run
    gpu-run
  ];
};
```

All scripts at once:

```nix
packages = [ pkgs.aleph.script.all-compiled ];
```
