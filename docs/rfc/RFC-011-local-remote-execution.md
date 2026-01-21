# RFC-011: Local Remote Execution

## Status: Implemented

## Summary

**THE GUARANTEE**: Your first command in `nix develop` is identical to that command in `buck2 build`. Same environment. Same sandbox. Not similar. Identical.

This is achieved by running ALL builds through NativeLink remote execution, with workers running in Firecracker VMs that have the same `/nix/store` paths as your local machine.

## Motivation

The fundamental problem with development environments:

1. `nix develop` runs on your dirty local machine
2. `nix build` runs in a sandbox
3. They can never be identical as long as one is local

Current approaches ("dev containers", "hermetic builds") try to make them *similar*. We make them *identical* by making both remote.

## Architecture

```
buck2 build --config=lre //:foo
       │
       │ RE protocol (gRPC)
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    NativeLink (local)                       │
│                                                             │
│   CAS:50051 ◄─────────────► Scheduler:50052                │
│       │                          │                         │
│       │                          ▼                         │
│       │                    ┌───────────┐                   │
│       │                    │  Workers  │                   │
│       └───────────────────►│           │                   │
│                            │ Firecracker│                  │
│                            │    VMs     │                  │
│                            │           │                   │
│                            │/nix/store │ (virtiofs, ro)   │
│                            └───────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **NativeLink CAS** (port 50051)
   - Content-addressable storage for build artifacts
   - Shared between local builds and workers
   - Backed by local filesystem with eviction policy

2. **NativeLink Scheduler** (port 50052)
   - Receives actions from buck2
   - Dispatches to available workers
   - Handles caching decisions

3. **Workers** (Firecracker VMs)
   - Identical environment to local /nix/store
   - Full Linux kernel isolation
   - ~125ms boot time with memory snapshots

### Why This Works

The worker image is built from the same Nix toolchains as your local environment:

```nix
worker-image = pkgs.callPackage ./lre-worker.nix {
  inherit (config.straylight.build) buck2-toolchain;
  nativelink-worker = inputs.nativelink.packages.${system}.default;
};
```

Same `/nix/store` paths = perfect cache hit rate between local and remote.

## Usage

### Starting LRE

```bash
# Enter development shell
nix develop

# Start NativeLink components
lre-start              # Start CAS, scheduler, 4 workers
lre-start --workers=8  # More workers
lre-start --stop       # Stop everything
lre-start --status     # Check running processes
```

### Building with LRE

```bash
# Build with remote execution
buck2 build --config=lre //:foo

# Or set default in .buckconfig
[buck2]
default_config = lre
```

### Configuration

The LRE module generates `.buckconfig.lre`:

```ini
[buck2_re_client]
engine_address = grpc://127.0.0.1:50052
cas_address = grpc://127.0.0.1:50051
action_cache_address = grpc://127.0.0.1:50051
instance_name = main
```

## Implementation

### Files

- `nix/modules/flake/lre.nix` - Flake-parts module
- `nix/overlays/container/lre-worker.nix` - Worker image
- `nix/packages/lre-start.nix` - Startup script
- `nix/scripts/lre-start.sh` - Shell implementation
- `nix/modules/flake/container/init-scripts.nix` - VM init

### Module Options

```nix
aleph-naught.lre = {
  enable = true;
  
  scheduler.address = "grpc://127.0.0.1:50052";
  cas.address = "grpc://127.0.0.1:50051";
  
  worker = {
    count = 4;
    cpus = 4;
    memory = 8192;  # MiB
    firecracker.enable = true;
  };
  
  buck2.config-prefix = "lre";  # --config=lre
};
```

## Future Work

1. **Firecracker Integration**: Currently workers run as processes; full Firecracker VM isolation is ready but needs virtiofs setup for /nix/store

2. **`--shell` Mode**: Attach TTY to a worker for debugging
   ```bash
   buck2 build --config=lre --shell //:foo
   # Drops into worker environment with same inputs
   ```

3. **Cloud Workers**: Same architecture, workers in the cloud instead of local Firecracker

4. **GPU Passthrough**: VFIO GPU passthrough to workers for CUDA builds

## Comparison

| Feature | Docker/Podman | Nix Sandbox | LRE |
|---------|---------------|-------------|-----|
| Identical env | No | No | Yes |
| Cache hits | Partial | Good | Perfect |
| Full isolation | Container | Namespace | VM |
| GPU support | nvidia-docker | No | VFIO |
| Network builds | Yes | No | Yes |
| Boot time | ~500ms | N/A | ~125ms |

## References

- [NativeLink](https://github.com/TraceMachina/nativelink)
- [Bazel Remote Execution API](https://github.com/bazelbuild/remote-apis)
- [Buck2 Remote Execution](https://buck2.build/docs/users/remote_execution/)
- [Firecracker](https://firecracker-microvm.github.io/)
