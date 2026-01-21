# nix/overlays/container/lre-worker.nix
#
# NativeLink LRE Worker Image
#
# This builds a worker image that can run buck2 actions with our
# Nix toolchains. The image contains:
#   - NativeLink worker binary
#   - All toolchain paths from buck2-toolchain
#   - Minimal runtime (busybox for /bin/sh)
#
# The key insight: the worker image has the SAME /nix/store paths
# as your local machine, so cache hits are perfect.
#
{
  lib,
  stdenvNoCC,
  buildEnv,
  writeText,
  writeShellScriptBin,
  busybox,
  coreutils,
  bash,
  gnugrep,
  gnused,
  gnutar,
  gzip,
  # From caller
  buck2-toolchain ? { },
  nativelink-worker,
}:
let
  # ════════════════════════════════════════════════════════════════════════════
  # Collect all toolchain paths
  # ════════════════════════════════════════════════════════════════════════════
  toolchainPaths = lib.filter (p: p != null) (
    lib.mapAttrsToList (
      _name: value:
      if lib.isString value && lib.hasPrefix "/nix/store" value then
        # Extract the store path (first component after /nix/store/)
        let
          parts = lib.splitString "/" value;
          # /nix/store/hash-name/... -> we want /nix/store/hash-name
          storePath = "/" + (lib.concatStringsSep "/" (lib.take 4 parts));
        in
        storePath
      else
        null
    ) buck2-toolchain
  );

  # Deduplicate
  uniquePaths = lib.unique toolchainPaths;

  # ════════════════════════════════════════════════════════════════════════════
  # Worker environment
  # ════════════════════════════════════════════════════════════════════════════
  workerEnv = buildEnv {
    name = "lre-worker-env";
    paths = [
      busybox
      coreutils
      bash
      gnugrep
      gnused
      gnutar
      gzip
      nativelink-worker
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # Worker configuration
  # ════════════════════════════════════════════════════════════════════════════
  workerConfig = writeText "worker-config.json" (
    builtins.toJSON {
      stores = [
        {
          memory = {
            eviction_policy = {
              max_bytes = 1000000000; # 1GB
            };
          };
        }
      ];
      workers = [
        {
          local = {
            worker_api_endpoint = {
              uri = "grpc://0.0.0.0:50061";
            };
            cas_fast_slow_store = {
              main = 0;
            };
            upload_action_result = {
              ac_store = 0;
            };
            work_directory = "/tmp/nativelink";
            # Platform properties for matching
            platform_properties = {
              OSFamily = "linux";
              container-image = "nix-lre-worker";
            };
          };
        }
      ];
    }
  );

  # ════════════════════════════════════════════════════════════════════════════
  # Entrypoint script
  # ════════════════════════════════════════════════════════════════════════════
  entrypoint = writeShellScriptBin "entrypoint" ''
    #!/bin/bash
    set -euo pipefail

    # Set up PATH with all toolchains
    export PATH="${workerEnv}/bin:$PATH"

    # Print toolchain paths for debugging
    echo "LRE Worker starting with toolchain paths:"
    ${lib.concatMapStringsSep "\n" (p: ''echo "  ${p}"'') uniquePaths}

    # Start the worker
    exec ${nativelink-worker}/bin/nativelink ${workerConfig}
  '';

in
stdenvNoCC.mkDerivation {
  pname = "lre-worker-image";
  version = "0.1.0";

  dontUnpack = true;

  buildPhase = ''
    runHook preBuild

    mkdir -p $out

    # Create a manifest of all required store paths
    cat > $out/store-paths.txt << 'EOF'
    ${lib.concatStringsSep "\n" uniquePaths}
    ${workerEnv}
    ${entrypoint}
    ${workerConfig}
    EOF

    # Create the entrypoint reference
    ln -s ${entrypoint}/bin/entrypoint $out/entrypoint

    # Create config reference
    ln -s ${workerConfig} $out/worker-config.json

    # Create environment reference
    ln -s ${workerEnv} $out/env

    runHook postBuild
  '';

  meta = {
    description = "NativeLink LRE worker with Nix toolchains";
    platforms = lib.platforms.linux;
  };
}
