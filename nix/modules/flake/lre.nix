# nix/modules/flake/lre.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // local remote execution //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "The sky above the port was the color of television,
#      tuned to a dead channel."
#
#                                                         — Neuromancer
#
# NativeLink integration for hermetic remote execution.
#
# THE GUARANTEE: Your first command in `nix develop` is identical to
# that command in `buck2 build`. Same environment. Same sandbox.
# Not similar. Identical.
#
# This is achieved by running ALL builds (local and "remote") in
# Firecracker VMs with identical Nix store mounts.
#
# ARCHITECTURE:
#
#   buck2 build //:foo
#          │
#          │ RE protocol (gRPC)
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │                    NativeLink (local)                       │
#   │                                                             │
#   │   scheduler ◄──────────────► CAS ◄──────────────► worker    │
#   │                                                     │       │
#   │                                          ┌──────────┘       │
#   │                                          ▼                  │
#   │                               ┌─────────────────────┐       │
#   │                               │   Firecracker VM    │       │
#   │                               │                     │       │
#   │                               │  /nix/store (ro)    │       │
#   │                               │  runs action        │       │
#   │                               │  returns outputs    │       │
#   │                               └─────────────────────┘       │
#   └─────────────────────────────────────────────────────────────┘
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
{
  config,
  lib,
  flake-parts-lib,
  ...
}:
let
  inherit (flake-parts-lib) mkPerSystemOption;
  cfg = config.aleph-naught.lre;
  buildCfg = config.aleph-naught.build;
in
{
  _class = "flake";

  # ════════════════════════════════════════════════════════════════════════════
  # OPTIONS
  # ════════════════════════════════════════════════════════════════════════════

  options.aleph-naught.lre = {
    enable = lib.mkEnableOption "Local Remote Execution via NativeLink";

    # ──────────────────────────────────────────────────────────────────────────
    # NativeLink Configuration
    # ──────────────────────────────────────────────────────────────────────────
    scheduler = {
      address = lib.mkOption {
        type = lib.types.str;
        default = "grpc://127.0.0.1:50052";
        description = "NativeLink scheduler address";
      };
    };

    cas = {
      address = lib.mkOption {
        type = lib.types.str;
        default = "grpc://127.0.0.1:50051";
        description = "NativeLink CAS (content-addressable storage) address";
      };
    };

    # ──────────────────────────────────────────────────────────────────────────
    # Worker Configuration
    # ──────────────────────────────────────────────────────────────────────────
    worker = {
      count = lib.mkOption {
        type = lib.types.int;
        default = 4;
        description = "Number of worker instances";
      };

      cpus = lib.mkOption {
        type = lib.types.int;
        default = 4;
        description = "CPUs per worker (Firecracker VM)";
      };

      memory = lib.mkOption {
        type = lib.types.int;
        default = 8192;
        description = "Memory per worker in MiB";
      };

      # Use Firecracker for isolation
      firecracker = {
        enable = lib.mkOption {
          type = lib.types.bool;
          default = true;
          description = "Run workers in Firecracker VMs for full isolation";
        };
      };
    };

    # ──────────────────────────────────────────────────────────────────────────
    # Buck2 Integration
    # ──────────────────────────────────────────────────────────────────────────
    buck2 = {
      config-prefix = lib.mkOption {
        type = lib.types.str;
        default = "lre";
        description = ''
          Prefix for buck2 config flags. With prefix "lre":
            buck2 build --config=lre //:foo
          Set to "" to enable LRE by default.
        '';
      };
    };
  };

  # Per-system options
  options.perSystem = mkPerSystemOption (
    { lib, ... }:
    {
      options.straylight.lre = {
        nativelink = lib.mkOption {
          type = lib.types.nullOr lib.types.package;
          default = null;
          description = "NativeLink package";
        };

        worker-image = lib.mkOption {
          type = lib.types.nullOr lib.types.package;
          default = null;
          description = "Worker OCI/Firecracker image";
        };

        buckconfig-re = lib.mkOption {
          type = lib.types.lines;
          default = "";
          description = "Buck2 RE configuration lines";
        };

        shellHook = lib.mkOption {
          type = lib.types.lines;
          default = "";
          description = "Shell hook for LRE setup";
        };
      };
    }
  );

  # ════════════════════════════════════════════════════════════════════════════
  # IMPLEMENTATION
  # ════════════════════════════════════════════════════════════════════════════

  config = lib.mkIf cfg.enable {
    perSystem =
      {
        pkgs,
        system,
        config,
        ...
      }:
      let
        inherit (pkgs.stdenv) isLinux;

        # Get NativeLink from inputs
        nativelink = inputs.nativelink.packages.${system}.default or null;
        nativelink-cli = inputs.nativelink.packages.${system}.native-cli or null;

        # Get build toolchain from build module
        buck2-toolchain = config.straylight.build.buck2-toolchain or { };

        # ────────────────────────────────────────────────────────────────────────
        # Worker Image
        # ────────────────────────────────────────────────────────────────────────
        # Build a worker image that includes all our toolchains
        worker-image = pkgs.callPackage ../../overlays/container/lre-worker.nix {
          inherit (config.straylight.build) buck2-toolchain;
          nativelink-worker = inputs.nativelink.packages.${system}.nativelink-worker or nativelink;
        };

        # ────────────────────────────────────────────────────────────────────────
        # Buck2 RE Configuration
        # ────────────────────────────────────────────────────────────────────────
        prefix = cfg.buck2.config-prefix;
        prefixedFlag = flag: if prefix == "" then flag else "${prefix}:${flag}";

        buckconfig-re = ''
          # NativeLink Remote Execution Configuration
          # Generated by aleph.lre module
          #
          # Usage: buck2 build ${lib.optionalString (prefix != "") "--config=${prefix} "}//:target

          [buck2_re_client]
          engine_address = ${cfg.scheduler.address}
          cas_address = ${cfg.cas.address}
          action_cache_address = ${cfg.cas.address}
          instance_name = main

          # Platform properties for worker matching
          [buck2_re_client.platform_properties]
          OSFamily = linux
          container-image = nix-lre-worker

          # Execution config
          build${lib.optionalString (prefix != "") ":${prefix}"} --remote_enabled=true
          build${lib.optionalString (prefix != "") ":${prefix}"} --remote_cache_enabled=true
        '';

        # ────────────────────────────────────────────────────────────────────────
        # Shell Hook
        # ────────────────────────────────────────────────────────────────────────
        lreShellHook = lib.optionalString isLinux ''
          # Generate lre.buckconfig for RE configuration
          cat > .buckconfig.lre << 'LRE_BUCKCONFIG_EOF'
          ${buckconfig-re}
          LRE_BUCKCONFIG_EOF
          echo "Generated .buckconfig.lre for NativeLink RE"

          # Append include to .buckconfig.local if not present
          if ! grep -q "buckconfig.lre" .buckconfig.local 2>/dev/null; then
            echo "" >> .buckconfig.local
            echo "# Include LRE configuration" >> .buckconfig.local
            echo "[include]" >> .buckconfig.local
            echo "    .buckconfig.lre" >> .buckconfig.local
          fi

          ${lib.optionalString (nativelink-cli != null) ''
            # Add native CLI to PATH
            export PATH="${nativelink-cli}/bin:$PATH"
          ''}
        '';

      in
      lib.mkIf isLinux {
        straylight.lre = {
          inherit nativelink worker-image buckconfig-re;
          shellHook = lreShellHook;
        };

        # Add to packages
        packages = lib.optionalAttrs (nativelink != null) {
          inherit nativelink;
          lre-worker-image = worker-image;
        };
      };

    # ──────────────────────────────────────────────────────────────────────────
    # Integrate with devshell
    # ──────────────────────────────────────────────────────────────────────────
    # The lre shellHook should run after the build shellHook
  };
}
