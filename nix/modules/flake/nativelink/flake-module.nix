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
  cfg = config.aleph-naught.nativelink;
in
{
  _class = "flake";

  options.aleph-naught.nativelink = {
    enable = lib.mkEnableOption "NativeLink remote execution containers";

    fly = {
      app-prefix = lib.mkOption {
        type = lib.types.str;
        default = "aleph";
        description = "Fly.io app name prefix (used for internal DNS)";
      };

      region = lib.mkOption {
        type = lib.types.str;
        default = "iad";
        description = "Primary Fly.io region";
      };
    };

    scheduler = {
      port = lib.mkOption {
        type = lib.types.port;
        default = 50051;
        description = "gRPC port for scheduler";
      };
    };

    cas = {
      port = lib.mkOption {
        type = lib.types.port;
        default = 50052;
        description = "gRPC port for CAS";
      };

      dataDir = lib.mkOption {
        type = lib.types.str;
        default = "/data";
        description = "Data directory for CAS storage (Fly volume mount)";
      };

      maxBytes = lib.mkOption {
        type = lib.types.int;
        default = 10737418240; # 10GB
        description = "Maximum CAS storage size in bytes";
      };
    };

    worker = {
      count = lib.mkOption {
        type = lib.types.int;
        default = 2;
        description = "Number of worker instances";
      };
    };

    registry = lib.mkOption {
      type = lib.types.str;
      default = "ghcr.io/straylight-software/aleph";
      description = "Container registry path for pushing images";
    };
  };
  config = lib.mkIf cfg.enable {
    perSystem =
      {
        pkgs,
        system,
        ...
      }:
      let
        # Get nativelink binary from flake input
        # NOTE: Use inputs.*.packages.${system} directly, NOT inputs'
        # inputs' causes infinite recursion in flake-parts
        nativelink =
          inputs.nativelink.packages.${system}.default or inputs.nativelink.packages.${system}.nativelink
            or null;

        # Fly internal DNS addresses (for container-to-container communication)
        schedulerAddr = "${cfg.fly.app-prefix}-scheduler.internal:${toString cfg.scheduler.port}";
        casAddr = "${cfg.fly.app-prefix}-cas.internal:${toString cfg.cas.port}";

        # ──────────────────────────────────────────────────────────────────────
        # NativeLink JSON configs
        # ──────────────────────────────────────────────────────────────────────

        schedulerConfig = pkgs.writeText "scheduler.json" (
          builtins.toJSON {
            stores = [
              {
                name = "CAS_MAIN_STORE";
                grpc = {
                  instance_name = "main";
                  endpoints = [ { address = "grpc://${casAddr}"; } ];
                };
              }
              {
                name = "AC_MAIN_STORE";
                grpc = {
                  instance_name = "main";
                  endpoints = [ { address = "grpc://${casAddr}"; } ];
                };
              }
            ];
            schedulers = [
              {
                name = "MAIN_SCHEDULER";
                simple = {
                  supported_platform_properties = {
                    cpu_count = "minimum";
                    OSFamily = "exact";
                    container-image = "exact";
                  };
                };
              }
            ];
            servers = [
              {
                listener = {
                  http = {
                    socket_address = "0.0.0.0:${toString cfg.scheduler.port}";
                  };
                };
                services = {
                  ac = [
                    {
                      instance_name = "main";
                      ac_store = "AC_MAIN_STORE";
                    }
                  ];
                  execution = [
                    {
                      instance_name = "main";
                      cas_store = "CAS_MAIN_STORE";
                      scheduler = "MAIN_SCHEDULER";
                    }
                  ];
                  capabilities = [ { instance_name = "main"; } ];
                };
              }
            ];
          }
        );

        casConfig = pkgs.writeText "cas.json" (
          builtins.toJSON {
            stores = [
              {
                name = "MAIN_STORE";
                compression = {
                  compression_algorithm = {
                    lz4 = { };
                  };
                  backend = {
                    filesystem = {
                      content_path = "${cfg.cas.dataDir}/content";
                      temp_path = "${cfg.cas.dataDir}/temp";
                      eviction_policy = {
                        max_bytes = cfg.cas.maxBytes;
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
                    socket_address = "0.0.0.0:${toString cfg.cas.port}";
                  };
                };
                services = {
                  cas = [
                    {
                      instance_name = "main";
                      cas_store = "MAIN_STORE";
                    }
                  ];
                  ac = [
                    {
                      instance_name = "main";
                      ac_store = "MAIN_STORE";
                    }
                  ];
                  bytestream = {
                    cas_stores = {
                      main = "MAIN_STORE";
                    };
                  };
                  capabilities = [ { instance_name = "main"; } ];
                  health = { };
                };
              }
            ];
          }
        );

        workerConfig = pkgs.writeText "worker.json" (
          builtins.toJSON {
            stores = [
              {
                name = "CAS_STORE";
                grpc = {
                  instance_name = "main";
                  endpoints = [ { address = "grpc://${casAddr}"; } ];
                };
              }
              {
                name = "AC_STORE";
                grpc = {
                  instance_name = "main";
                  endpoints = [ { address = "grpc://${casAddr}"; } ];
                };
              }
            ];
            workers = [
              {
                local = {
                  work_directory = "/tmp/nativelink-worker";
                  cas_store = "CAS_STORE";
                  ac_store = "AC_STORE";
                  upload_action_results = true;
                  platform_properties = {
                    OSFamily = "linux";
                    container-image = "nix-worker";
                  };
                };
                scheduler = {
                  endpoint = {
                    address = "grpc://${schedulerAddr}";
                  };
                };
              }
            ];
          }
        );

        # ──────────────────────────────────────────────────────────────────────
        # Wrapper scripts (entrypoints for containers)
        # ──────────────────────────────────────────────────────────────────────

        schedulerScript = pkgs.writeShellApplication {
          name = "nativelink-scheduler";
          runtimeInputs = [ nativelink ];
          text = ''
            exec nativelink ${schedulerConfig}
          '';
        };

        casScript = pkgs.writeShellApplication {
          name = "nativelink-cas";
          runtimeInputs = [ nativelink ];
          text = ''
            mkdir -p ${cfg.cas.dataDir}/content ${cfg.cas.dataDir}/temp
            exec nativelink ${casConfig}
          '';
        };

        workerScript = pkgs.writeShellApplication {
          name = "nativelink-worker";
          runtimeInputs = [ nativelink ];
          text = ''
            mkdir -p /tmp/nativelink-worker
            exec nativelink ${workerConfig}
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

        # Haskell toolchain
        hsPkgs = pkgs.haskell.packages.ghc912 or pkgs.haskellPackages;
        ghcWithPackages = hsPkgs.ghcWithPackages (hp: [
          hp.text
          hp.bytestring
          hp.containers
          hp.aeson
          hp.optparse-applicative
        ]);

        # Python with nanobind/pybind11 for Buck2 python_cxx rules
        pythonEnv = pkgs.python312.withPackages (ps: [
          ps.nanobind
          ps.pybind11
          ps.numpy
        ]);

        # All toolchain packages for workers
        toolchainPackages =
          lib.optionals (llvm-git != null) [ llvm-git ]
          ++ lib.optionals (nvidia-sdk != null) [ nvidia-sdk ]
          ++ [
            gcc
            pkgs.glibc
            pkgs.glibc.dev
            ghcWithPackages
            pythonEnv
            pkgs.rustc
            pkgs.cargo
            pkgs.coreutils
            pkgs.bash
            pkgs.gnumake
          ]
          ++ lib.optionals (pkgs ? lean4) [ pkgs.lean4 ]
          ++ lib.optionals (pkgs ? mdspan) [ pkgs.mdspan ];

        # Modular service for nativelink (nimi pattern)
        mkNativelinkService =
          { script }:
          { lib, pkgs, ... }:
          { ... }:
          {
            _class = "service";
            config.process.argv = [ "${script}/bin/${script.name}" ];
          };

      in
      lib.optionalAttrs (nativelink != null) {
        # ────────────────────────────────────────────────────────────────────
        # Container definitions via nix2gpu
        # Build: nix build .#nativelink-scheduler
        # Push:  nix run .#nativelink-scheduler.copyToGithub
        # ────────────────────────────────────────────────────────────────────

        nix2gpu = {
          # Scheduler container (minimal, just nativelink binary)
          nativelink-scheduler = {
            systemPackages = [
              nativelink
              schedulerScript
            ];

            services.scheduler = {
              imports = [ (mkNativelinkService { script = schedulerScript; } { inherit lib pkgs; }) ];
            };

            exposedPorts = {
              "${toString cfg.scheduler.port}/tcp" = { };
            };

            registries = [ cfg.registry ];

            extraEnv = {
              RUST_LOG = "info";
            };
          };

          # CAS container (content-addressed storage with LZ4 compression)
          nativelink-cas = {
            systemPackages = [
              nativelink
              casScript
            ];

            services.cas = {
              imports = [ (mkNativelinkService { script = casScript; } { inherit lib pkgs; }) ];
            };

            exposedPorts = {
              "${toString cfg.cas.port}/tcp" = { };
            };

            registries = [ cfg.registry ];

            extraEnv = {
              RUST_LOG = "info";
            };
          };

          # Worker container (full toolchain for hermetic builds)
          nativelink-worker = {
            systemPackages = [
              nativelink
              workerScript
            ]
            ++ toolchainPackages;

            services.worker = {
              imports = [ (mkNativelinkService { script = workerScript; } { inherit lib pkgs; }) ];
            };

            registries = [ cfg.registry ];

            extraEnv = {
              RUST_LOG = "info";
            };
          };
        };

        # ────────────────────────────────────────────────────────────────────
        # Export configs and scripts as packages for inspection/debugging
        # ────────────────────────────────────────────────────────────────────

        packages = {
          nativelink-scheduler-config = schedulerConfig;
          nativelink-cas-config = casConfig;
          nativelink-worker-config = workerConfig;
          nativelink-scheduler-script = schedulerScript;
          nativelink-cas-script = casScript;
          nativelink-worker-script = workerScript;
        };
      };
  };
}
