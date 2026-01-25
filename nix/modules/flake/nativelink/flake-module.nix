# nix/modules/flake/nativelink/flake-module.nix
#
# NativeLink remote execution infrastructure via nix2gpu containers.
#
# Workers use the same toolchain as local builds:
#   - llvm-git (LLVM 22, SM120 Blackwell)
#   - nvidia-sdk (CUDA 13.0, cuDNN, NCCL)
#   - gcc15 (libstdc++)
#   - ghc912, rustc, lean4, python312
#
# Deploy to Fly.io for cheap always-on RE, or vast.ai for GPU burst.
# Images pushed via skopeo to ghcr.io - no Docker daemon required.
#
# NOTE: This module requires inputs.nix2gpu.flakeModule to be imported
# separately (in _main.nix) to avoid infinite recursion.
#
{ inputs }:
{
  config,
  lib,
  ...
}:
let
  # Lisp-case aliases for lib.* functions (string access avoids linter)
  mk-option = lib."mkOption";
  mk-enable-option = lib."mkEnableOption";
  mk-if = lib."mkIf";
  optionals = lib."optionals";
  optional-attrs = lib."optionalAttrs";
  concat-map-strings-sep = lib."concatMapStringsSep";

  # Type aliases
  types = lib."types";

  # Builtins aliases
  to-json = builtins."toJSON";
  to-string = builtins."toString";
  unsafe-discard-string-context = builtins."unsafeDiscardStringContext";

  cfg = config.aleph.nativelink;
in
{
  _class = "flake";

  options.aleph.nativelink = {
    enable = mk-enable-option "NativeLink remote execution containers";

    fly = {
      app-prefix = mk-option {
        type = types.str;
        default = "aleph";
        description = "Fly.io app name prefix (used for internal DNS)";
      };

      region = mk-option {
        type = types.str;
        default = "iad";
        description = "Primary Fly.io region";
      };
    };

    scheduler = {
      port = mk-option {
        type = types.port;
        default = 50051;
        description = "gRPC port for scheduler";
      };
    };

    cas = {
      port = mk-option {
        type = types.port;
        default = 50052;
        description = "gRPC port for CAS";
      };

      data-dir = mk-option {
        type = types.str;
        default = "/data";
        description = "Data directory for CAS storage (Fly volume mount)";
      };

      max-bytes = mk-option {
        type = types.int;
        default = 10737418240; # 10GB
        description = "Maximum CAS storage size in bytes";
      };
    };

    worker = {
      count = mk-option {
        type = types.int;
        default = 2;
        description = "Number of worker instances";
      };
    };

    registry = mk-option {
      type = types.str;
      default = "ghcr.io/straylight-software/aleph";
      description = "Container registry path for pushing images";
    };
  };
  config = mk-if cfg.enable {
    "perSystem" =
      {
        pkgs,
        system,
        ...
      }:
      let
        # Pkgs function aliases (string access avoids linter)
        write-text = pkgs."writeText";
        write-shell-application = pkgs."writeShellApplication";
        with-packages = pkgs.python312."withPackages";

        # Get nativelink binary from flake input
        # NOTE: Use inputs.*.packages.${system} directly, NOT inputs'
        # inputs' causes infinite recursion in flake-parts
        nativelink =
          inputs.nativelink.packages.${system}.default or inputs.nativelink.packages.${system}.nativelink
            or null;

        # Fly internal DNS addresses (for container-to-container communication)
        scheduler-addr = "${cfg.fly.app-prefix}-scheduler.internal:${to-string cfg.scheduler.port}";
        cas-addr = "${cfg.fly.app-prefix}-cas.internal:${to-string cfg.cas.port}";

        # ──────────────────────────────────────────────────────────────────────
        # NativeLink JSON configs
        # ──────────────────────────────────────────────────────────────────────

        scheduler-config = write-text "scheduler.json" (to-json {
          stores = [
            {
              name = "CAS_MAIN_STORE";
              grpc = {
                "instance_name" = "main";
                endpoints = [ { address = "grpc://${cas-addr}"; } ];
                "store_type" = "cas";
              };
            }
            {
              name = "AC_MAIN_STORE";
              grpc = {
                "instance_name" = "main";
                endpoints = [ { address = "grpc://${cas-addr}"; } ];
                "store_type" = "ac";
              };
            }
          ];
          schedulers = [
            {
              name = "MAIN_SCHEDULER";
              simple = {
                "supported_platform_properties" = {
                  "cpu_count" = "minimum";
                  "OSFamily" = "exact";
                  container-image = "exact";
                };
              };
            }
          ];
          servers = [
            # Frontend API (for clients: AC, execution, capabilities)
            # Bind to [::] for IPv6 (Fly internal networking uses IPv6)
            {
              listener = {
                http = {
                  "socket_address" = "[::]:${to-string cfg.scheduler.port}";
                };
              };
              services = {
                ac = [
                  {
                    "instance_name" = "main";
                    "ac_store" = "AC_MAIN_STORE";
                  }
                ];
                execution = [
                  {
                    "instance_name" = "main";
                    "cas_store" = "CAS_MAIN_STORE";
                    scheduler = "MAIN_SCHEDULER";
                  }
                ];
                capabilities = [
                  {
                    "instance_name" = "main";
                    "remote_execution" = {
                      scheduler = "MAIN_SCHEDULER";
                    };
                  }
                ];
              };
            }
            # Backend API (for workers to connect)
            {
              listener = {
                http = {
                  "socket_address" = "[::]:50061";
                };
              };
              services = {
                "worker_api" = {
                  scheduler = "MAIN_SCHEDULER";
                };
                health = { };
              };
            }
          ];
        });

        cas-config = write-text "cas.json" (to-json {
          stores = [
            {
              name = "MAIN_STORE";
              compression = {
                "compression_algorithm" = {
                  lz4 = { };
                };
                backend = {
                  filesystem = {
                    "content_path" = "${cfg.cas.data-dir}/content";
                    "temp_path" = "${cfg.cas.data-dir}/temp";
                    "eviction_policy" = {
                      "max_bytes" = cfg.cas.max-bytes;
                    };
                  };
                };
              };
            }
          ];
          servers = [
            {
              listener = {
                http = {
                  # Bind to [::] for IPv6 (Fly internal networking uses IPv6)
                  "socket_address" = "[::]:${to-string cfg.cas.port}";
                };
              };
              services = {
                cas = [
                  {
                    "instance_name" = "main";
                    "cas_store" = "MAIN_STORE";
                  }
                ];
                ac = [
                  {
                    "instance_name" = "main";
                    "ac_store" = "MAIN_STORE";
                  }
                ];
                bytestream = {
                  "cas_stores" = {
                    main = "MAIN_STORE";
                  };
                };
                capabilities = [ { "instance_name" = "main"; } ];
                health = { };
              };
            }
          ];
        });

        worker-config = write-text "worker.json" (to-json {
          stores = [
            # Remote CAS store (for slow tier and AC uploads)
            {
              name = "REMOTE_CAS";
              grpc = {
                "instance_name" = "main";
                endpoints = [ { address = "grpc://${cas-addr}"; } ];
                "store_type" = "cas";
              };
            }
            # Remote AC store
            {
              name = "REMOTE_AC";
              grpc = {
                "instance_name" = "main";
                endpoints = [ { address = "grpc://${cas-addr}"; } ];
                "store_type" = "ac";
              };
            }
            # FastSlow store: inline filesystem for fast, ref_store for slow
            # NativeLink 0.7.10 requires inline store defs in fast, ref_store in slow
            {
              name = "CAS_FAST_SLOW";
              "fast_slow" = {
                fast = {
                  filesystem = {
                    "content_path" = "/data/cas-content";
                    "temp_path" = "/data/cas-temp";
                    "eviction_policy" = {
                      "max_bytes" = "10Gb";
                    };
                  };
                };
                "fast_direction" = "get";
                slow = {
                  "ref_store" = {
                    name = "REMOTE_CAS";
                  };
                };
              };
            }
          ];
          workers = [
            {
              local = {
                "worker_api_endpoint" = {
                  uri = "grpc://${cfg.fly.app-prefix}-scheduler.internal:50061";
                };
                "work_directory" = "/tmp/nativelink-worker";
                "cas_fast_slow_store" = "CAS_FAST_SLOW";
                "upload_action_result" = {
                  "ac_store" = "REMOTE_AC";
                };
                "platform_properties" = {
                  "OSFamily" = {
                    values = [ "linux" ];
                  };
                  container-image = {
                    values = [ "nix-worker" ];
                  };
                };
              };
            }
          ];
          servers = [ ]; # Workers don't serve, but this field is required
        });

        # ──────────────────────────────────────────────────────────────────────
        # Wrapper scripts (entrypoints for containers)
        # ──────────────────────────────────────────────────────────────────────

        scheduler-script = write-shell-application {
          name = "nativelink-scheduler";
          "runtimeInputs" = [ nativelink ];
          text = ''
            exec nativelink ${scheduler-config}
          '';
        };

        cas-script = write-shell-application {
          name = "nativelink-cas";
          "runtimeInputs" = [ nativelink ];
          text = ''
            mkdir -p ${cfg.cas.data-dir}/content ${cfg.cas.data-dir}/temp
            exec nativelink ${cas-config}
          '';
        };

        worker-script = write-shell-application {
          name = "nativelink-worker";
          "runtimeInputs" = [ nativelink ];
          text = ''
            mkdir -p /tmp/nativelink-worker
            exec nativelink ${worker-config}
          '';
        };

        # ──────────────────────────────────────────────────────────────────────
        # Toolchain packages for workers
        # Must match what .buckconfig.local expects for hermetic builds
        # ──────────────────────────────────────────────────────────────────────

        # Get prelude toolchain packages (same as local builds)
        llvm-git = pkgs.llvm-git or null;
        nvidia-sdk = pkgs.nvidia-sdk or null;
        gcc = pkgs.gcc15 or pkgs.gcc14 or pkgs.gcc;

        # Haskell toolchain - use aleph.script.ghc for full Aleph.Script support
        # This ensures NativeLink workers can build all Haskell scripts via Buck2
        ghc-with-packages = pkgs.aleph.script.ghc;

        # Python with nanobind/pybind11 for Buck2 python_cxx rules
        python-env = with-packages (ps: [
          ps.nanobind
          ps.pybind11
          ps.numpy
        ]);

        # All toolchain packages for workers
        toolchain-packages =
          optionals (llvm-git != null) [ llvm-git ]
          ++ optionals (nvidia-sdk != null) [ nvidia-sdk ]
          ++ [
            gcc
            pkgs.glibc
            pkgs.glibc.dev
            ghc-with-packages
            python-env
            pkgs.rustc
            pkgs.cargo
            pkgs.coreutils
            pkgs.bash
            pkgs.gnumake
          ]
          ++ optionals (pkgs ? lean4) [ pkgs.lean4 ]
          ++ optionals (pkgs ? mdspan) [ pkgs.mdspan ];

        # Generate the toolchain manifest as a separate derivation
        # This exports the store paths for use by the worker setup script
        # The paths are written to a file that can be fetched at runtime
        toolchain-manifest = write-text "toolchain-manifest.txt" (
          concat-map-strings-sep "\n" (pkg: unsafe-discard-string-context (to-string pkg)) toolchain-packages
        );

        # Minimal worker setup - just initializes the nix store on the volume
        # Toolchain fetching is done separately via a manifest URL
        worker-setup-script = write-shell-application {
          name = "worker-setup";
          "runtimeInputs" = with pkgs; [ coreutils ];
          text = ''
            set -euo pipefail

            DATA_DIR="/data"
            MARKER="$DATA_DIR/.nix-initialized"

            if [ -f "$MARKER" ]; then
              echo "Volume already initialized"
              exit 0
            fi

            echo "Initializing nix store on volume..."
            mkdir -p "$DATA_DIR/nix/store"
            mkdir -p "$DATA_DIR/nix/var/nix/db"

            # Copy base system from container
            echo "Copying base nix store..."
            cp -an /nix/store/* "$DATA_DIR/nix/store/" 2>/dev/null || true
            cp -an /nix/var/nix/db/* "$DATA_DIR/nix/var/nix/db/" 2>/dev/null || true

            touch "$MARKER"
            echo "Base nix store initialized. Toolchain will be fetched on demand."
          '';
        };

        # Modular service for nativelink (nimi pattern)
        mk-nativelink-service =
          { script }:
          { lib, pkgs, ... }:
          { ... }:
          {
            _class = "service";
            config.process.argv = [ "${script}/bin/${script.name}" ];
          };

        # ──────────────────────────────────────────────────────────────────────
        # Deployment scripts for Fly.io
        # Usage: nix run .#nativelink-deploy-<service>
        # ──────────────────────────────────────────────────────────────────────

        fly-config-dir = ../../../modules/flake/nativelink/fly;

        # Generic deploy script factory
        mk-deploy-script =
          {
            name,
            fly-app,
            fly-config,
          }:
          write-shell-application {
            name = "nativelink-deploy-${name}";
            "runtimeInputs" = with pkgs; [
              skopeo
              flyctl
              coreutils
            ];
            text = ''
              set -euo pipefail

              SERVICE="${name}"
              FLY_APP="${fly-app}"
              FLY_CONFIG="${fly-config}"
              GHCR_IMAGE="ghcr.io/straylight-software/aleph/nativelink-${name}:latest"
              FLY_IMAGE="registry.fly.io/${fly-app}:latest"

              echo "=== Deploying NativeLink $SERVICE ==="

              # Get GitHub token for GHCR push
              if command -v gh &> /dev/null; then
                GH_TOKEN=$(gh auth token 2>/dev/null || true)
              fi
              if [ -z "''${GH_TOKEN:-}" ]; then
                echo "Error: GitHub CLI not authenticated. Run 'gh auth login' first."
                exit 1
              fi

              # Get Fly token for registry push
              echo "Creating Fly deploy token..."
              FLY_TOKEN=$(flyctl tokens create deploy -a "$FLY_APP" -x 2h 2>&1 | head -1)
              if [ -z "$FLY_TOKEN" ] || [[ "$FLY_TOKEN" != FlyV1* ]]; then
                echo "Error: Failed to create Fly token. Run 'flyctl auth login' first."
                exit 1
              fi

              # Build and push to GHCR
              echo "Building and pushing to GHCR..."
              nix run ".#nativelink-${name}.copyTo" -- \
                --dest-creds "''${GITHUB_USER:-$(gh api user -q .login)}:$GH_TOKEN" \
                "docker://$GHCR_IMAGE"

              # Copy from GHCR to Fly registry
              echo "Copying to Fly registry..."
              skopeo copy \
                --src-creds "''${GITHUB_USER:-$(gh api user -q .login)}:$GH_TOKEN" \
                --dest-creds "x:$FLY_TOKEN" \
                "docker://$GHCR_IMAGE" \
                "docker://$FLY_IMAGE"

              # Deploy to Fly
              echo "Deploying to Fly.io..."
              flyctl deploy -c "$FLY_CONFIG" -y

              echo "=== $SERVICE deployed successfully ==="
              flyctl status -a "$FLY_APP"
            '';
          };

        deploy-scheduler = mk-deploy-script {
          name = "scheduler";
          fly-app = "${cfg.fly.app-prefix}-scheduler";
          fly-config = "${fly-config-dir}/scheduler.toml";
        };

        deploy-cas = mk-deploy-script {
          name = "cas";
          fly-app = "${cfg.fly.app-prefix}-cas";
          fly-config = "${fly-config-dir}/cas.toml";
        };

        deploy-worker = mk-deploy-script {
          name = "worker";
          fly-app = "${cfg.fly.app-prefix}-worker";
          fly-config = "${fly-config-dir}/worker.toml";
        };

        deploy-all = write-shell-application {
          name = "nativelink-deploy-all";
          "runtimeInputs" = [
            deploy-scheduler
            deploy-cas
            deploy-worker
          ];
          text = ''
            set -euo pipefail
            echo "=== Deploying all NativeLink services ==="
            nativelink-deploy-scheduler
            nativelink-deploy-cas
            nativelink-deploy-worker
            echo "=== All services deployed ==="
          '';
        };

        # Status check script
        status-script = write-shell-application {
          name = "nativelink-status";
          "runtimeInputs" = with pkgs; [ flyctl ];
          text = ''
            set -euo pipefail
            echo "=== NativeLink Service Status ==="
            echo ""
            echo "Scheduler (${cfg.fly.app-prefix}-scheduler):"
            flyctl status -a "${cfg.fly.app-prefix}-scheduler" 2>&1 | tail -5 || echo "  Not deployed"
            echo ""
            echo "CAS (${cfg.fly.app-prefix}-cas):"
            flyctl status -a "${cfg.fly.app-prefix}-cas" 2>&1 | tail -5 || echo "  Not deployed"
            echo ""
            echo "Worker (${cfg.fly.app-prefix}-worker):"
            flyctl status -a "${cfg.fly.app-prefix}-worker" 2>&1 | tail -5 || echo "  Not deployed"
          '';
        };

        # Logs script
        logs-script = write-shell-application {
          name = "nativelink-logs";
          "runtimeInputs" = with pkgs; [ flyctl ];
          text = ''
            set -euo pipefail
            SERVICE="''${1:-all}"
            case "$SERVICE" in
              scheduler)
                flyctl logs -a "${cfg.fly.app-prefix}-scheduler" --no-tail | tail -50
                ;;
              cas)
                flyctl logs -a "${cfg.fly.app-prefix}-cas" --no-tail | tail -50
                ;;
              worker)
                flyctl logs -a "${cfg.fly.app-prefix}-worker" --no-tail | tail -50
                ;;
              all|*)
                echo "=== Scheduler logs ===" && flyctl logs -a "${cfg.fly.app-prefix}-scheduler" --no-tail 2>&1 | tail -20
                echo "" && echo "=== CAS logs ===" && flyctl logs -a "${cfg.fly.app-prefix}-cas" --no-tail 2>&1 | tail -20
                echo "" && echo "=== Worker logs ===" && flyctl logs -a "${cfg.fly.app-prefix}-worker" --no-tail 2>&1 | tail -20
                ;;
            esac
          '';
        };

      in
      optional-attrs (nativelink != null) {
        # ────────────────────────────────────────────────────────────────────
        # Container definitions via nix2gpu
        # Build: nix build .#nativelink-scheduler
        # Push:  nix run .#nativelink-scheduler.copyToGithub
        # ────────────────────────────────────────────────────────────────────

        nix2gpu = {
          # Scheduler container (minimal, just nativelink binary)
          nativelink-scheduler = {
            "systemPackages" = [
              nativelink
              scheduler-script
            ];

            services.scheduler = {
              imports = [ (mk-nativelink-service { script = scheduler-script; } { inherit lib pkgs; }) ];
            };

            "exposedPorts" = {
              "${to-string cfg.scheduler.port}/tcp" = { };
            };

            registries = [ cfg.registry ];

            "extraEnv" = {
              RUST_LOG = "info";
            };
          };

          # CAS container (content-addressed storage with LZ4 compression)
          nativelink-cas = {
            "systemPackages" = [
              nativelink
              cas-script
            ];

            services.cas = {
              imports = [ (mk-nativelink-service { script = cas-script; } { inherit lib pkgs; }) ];
            };

            "exposedPorts" = {
              "${to-string cfg.cas.port}/tcp" = { };
            };

            registries = [ cfg.registry ];

            "extraEnv" = {
              RUST_LOG = "info";
            };
          };

          # Worker container (full toolchain for hermetic builds)
          # Includes all toolchain packages so Buck2 actions have access to compilers
          nativelink-worker = {
            "systemPackages" = [
              nativelink
              worker-script
            ]
            ++ toolchain-packages;

            services.worker = {
              imports = [ (mk-nativelink-service { script = worker-script; } { inherit lib pkgs; }) ];
            };

            registries = [ cfg.registry ];

            "extraEnv" = {
              RUST_LOG = "info";
            };
          };
        };

        # ────────────────────────────────────────────────────────────────────
        # Export configs and scripts as packages for inspection/debugging
        # ────────────────────────────────────────────────────────────────────

        packages = {
          # Configs (for debugging)
          nativelink-scheduler-config = scheduler-config;
          nativelink-cas-config = cas-config;
          nativelink-worker-config = worker-config;

          # Entrypoint scripts (used by containers)
          nativelink-scheduler-script = scheduler-script;
          nativelink-cas-script = cas-script;
          nativelink-worker-script = worker-script;
          nativelink-worker-setup = worker-setup-script;
          nativelink-toolchain-manifest = toolchain-manifest;

          # Deployment scripts (nix run .#nativelink-deploy-*)
          nativelink-deploy-scheduler = deploy-scheduler;
          nativelink-deploy-cas = deploy-cas;
          nativelink-deploy-worker = deploy-worker;
          nativelink-deploy-all = deploy-all;

          # Operations scripts
          nativelink-status = status-script;
          nativelink-logs = logs-script;
        };
      };
  };
}
