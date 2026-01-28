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
  read-file = builtins."readFile";
  replace-strings = builtins."replaceStrings";
  unsafe-discard-string-context = builtins."unsafeDiscardStringContext";

  # Script directory
  scripts-dir = ./scripts;

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

    # ──────────────────────────────────────────────────────────────────────────
    # Builder: dedicated nix build machine on Fly
    # SSH in, run nix build, push images. Your laptop stays cool.
    # ──────────────────────────────────────────────────────────────────────────
    builder = {
      enable = mk-option {
        type = types.bool;
        default = true;
        description = "Deploy a dedicated nix builder on Fly";
      };

      cpus = mk-option {
        type = types.int;
        default = 16;
        description = "CPUs for builder (performance cores)";
      };

      memory = mk-option {
        type = types.str;
        default = "32gb";
        description = "RAM for builder";
      };

      volume-size = mk-option {
        type = types.str;
        default = "200gb";
        description = "Nix store volume size";
      };
    };

    # ──────────────────────────────────────────────────────────────────────────
    # Scheduler: coordinates work, routes actions to workers
    # Stateless, minimal resources needed
    # ──────────────────────────────────────────────────────────────────────────
    scheduler = {
      port = mk-option {
        type = types.port;
        default = 50051;
        description = "gRPC port for scheduler";
      };

      cpus = mk-option {
        type = types.int;
        default = 2;
        description = "CPUs for scheduler";
      };

      memory = mk-option {
        type = types.str;
        default = "2gb";
        description = "RAM for scheduler";
      };
    };

    # ──────────────────────────────────────────────────────────────────────────
    # CAS: content-addressed storage with LZ4 compression
    # Hot path for all blob transfers - size aggressively
    #
    # With R2 backend enabled, uses fast_slow store:
    #   - fast: local filesystem (eviction-based LRU)
    #   - slow: Cloudflare R2 (S3-compatible, persistent)
    # ──────────────────────────────────────────────────────────────────────────
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
        default = 500 * 1024 * 1024 * 1024; # 500GB
        description = "Maximum CAS storage size in bytes (local fast tier)";
      };

      cpus = mk-option {
        type = types.int;
        default = 4;
        description = "CPUs for CAS server";
      };

      memory = mk-option {
        type = types.str;
        default = "8gb";
        description = "RAM for CAS (more = better caching)";
      };

      volume-size = mk-option {
        type = types.str;
        default = "500gb";
        description = "CAS storage volume size";
      };

      # ────────────────────────────────────────────────────────────────────────
      # R2 Backend: Cloudflare R2 as slow tier for persistent global storage
      # ────────────────────────────────────────────────────────────────────────
      r2 = {
        enable = mk-option {
          type = types.bool;
          default = false;
          description = "Enable Cloudflare R2 as slow tier backend";
        };

        bucket = mk-option {
          type = types.str;
          default = "nativelink-cas";
          description = "R2 bucket name";
        };

        endpoint = mk-option {
          type = types.str;
          default = "https://bb63fff6a3d0856513474ee20860a81a.r2.cloudflarestorage.com";
          description = "R2 S3-compatible endpoint URL";
        };

        key-prefix = mk-option {
          type = types.str;
          default = "cas/";
          description = "Key prefix for objects in bucket";
        };

        # Credentials loaded from environment:
        #   AWS_ACCESS_KEY_ID
        #   AWS_SECRET_ACCESS_KEY
        # Set via Fly secrets or local env
      };
    };

    # ──────────────────────────────────────────────────────────────────────────
    # Workers: execute build actions
    # These do the actual compilation. Go big.
    # 8x performance-16 = 128 cores total
    # ──────────────────────────────────────────────────────────────────────────
    worker = {
      count = mk-option {
        type = types.int;
        default = 8;
        description = "Number of worker instances";
      };

      cpus = mk-option {
        type = types.int;
        default = 16;
        description = "CPUs per worker (Fly performance cores)";
      };

      memory = mk-option {
        type = types.str;
        default = "32gb";
        description = "RAM per worker";
      };

      cpu-kind = mk-option {
        type = types.enum [
          "shared"
          "performance"
        ];
        default = "performance";
        description = "Fly CPU type (performance = dedicated)";
      };

      volume-size = mk-option {
        type = types.str;
        default = "100gb";
        description = "Persistent volume size for nix store";
      };
    };

    registry = mk-option {
      type = types.str;
      default = "ghcr.io/straylight-software/aleph";
      description = "Container registry path for pushing images";
    };

    # ──────────────────────────────────────────────────────────────────────────
    # Nix Proxy: caching HTTP proxy for build-time fetches
    # Routes cargo, npm, pip, etc. through mitmproxy with content-addressed cache.
    # Workers set HTTP_PROXY to point at this on Fly internal network.
    # ──────────────────────────────────────────────────────────────────────────
    nix-proxy = {
      enable = mk-option {
        type = types.bool;
        default = true;
        description = "Deploy a caching HTTP proxy for build-time fetches";
      };

      port = mk-option {
        type = types.port;
        default = 8080;
        description = "HTTP proxy port";
      };

      cpus = mk-option {
        type = types.int;
        default = 2;
        description = "CPUs for proxy";
      };

      memory = mk-option {
        type = types.str;
        default = "4gb";
        description = "RAM for proxy";
      };

      volume-size = mk-option {
        type = types.str;
        default = "100gb";
        description = "Cache volume size";
      };

      allowlist = mk-option {
        type = types.listOf types.str;
        default = [
          # Nix caches
          "cache.nixos.org"
          "nix-community.cachix.org"
          # Source forges
          "github.com"
          "githubusercontent.com"
          "gitlab.com"
          "bitbucket.org"
          # Package registries
          "crates.io"
          "static.crates.io"
          "index.crates.io"
          "pypi.org"
          "files.pythonhosted.org"
          "registry.npmjs.org"
          "hackage.haskell.org"
        ];
        description = "Domain allowlist for proxy (empty = allow all)";
      };
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
          inputs.nativelink.packages.${system}.default or inputs.nativelink.packages.${system}.nativelink;

        # Fly internal DNS addresses (for container-to-container communication)
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

        # CAS configuration - uses fast_slow store when R2 is enabled
        # Fast tier: local filesystem with LRU eviction
        # Slow tier: Cloudflare R2 (S3-compatible) for persistent global storage
        cas-config = write-text "cas.json" (to-json {
          stores =
            if cfg.cas.r2.enable then
              [
                # Local filesystem as fast tier (LRU cache)
                {
                  name = "LOCAL_FILESYSTEM";
                  compression = {
                    "compression_algorithm".lz4 = { };
                    backend.filesystem = {
                      "content_path" = "${cfg.cas.data-dir}/content";
                      "temp_path" = "${cfg.cas.data-dir}/temp";
                      "eviction_policy"."max_bytes" = cfg.cas.max-bytes;
                    };
                  };
                }
                # R2 as slow tier (persistent S3-compatible storage)
                {
                  name = "R2_STORE";
                  "experimental_s3_store" = {
                    region = "auto"; # R2 uses "auto" for region
                    inherit (cfg.cas.r2) bucket;
                    "key_prefix" = cfg.cas.r2.key-prefix;
                    retry = {
                      "max_retries" = 6;
                      delay = 0.3;
                      jitter = 0.5;
                    };
                    # Credentials from environment: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
                  }
                  // optional-attrs (cfg.cas.r2.endpoint != "") {
                    "endpoint_url" = cfg.cas.r2.endpoint;
                  };
                }
                # Fast/slow composite store
                {
                  name = "MAIN_STORE";
                  "fast_slow" = {
                    fast."ref_store".name = "LOCAL_FILESYSTEM";
                    slow."ref_store".name = "R2_STORE";
                  };
                }
              ]
            else
              [
                # Simple filesystem store (no R2)
                {
                  name = "MAIN_STORE";
                  compression = {
                    "compression_algorithm".lz4 = { };
                    backend.filesystem = {
                      "content_path" = "${cfg.cas.data-dir}/content";
                      "temp_path" = "${cfg.cas.data-dir}/temp";
                      "eviction_policy"."max_bytes" = cfg.cas.max-bytes;
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
        # Use explicit ? checks for optional packages
        llvm-git = if pkgs ? llvm-git then pkgs.llvm-git else null;
        nvidia-sdk = if pkgs ? nvidia-sdk then pkgs.nvidia-sdk else null;
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

        # Worker setup - fetches toolchain from cache.nixos.org on first boot
        # Toolchain paths are baked into the manifest at build time
        worker-setup-script = write-shell-application {
          name = "worker-setup";
          "runtimeInputs" = with pkgs; [
            coreutils
            nix
            cacert
          ];
          "runtimeEnv" = {
            NIX_SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
          };
          text =
            replace-strings
              [ "@toolchainPaths@" ]
              [
                (concat-map-strings-sep " " (pkg: unsafe-discard-string-context (to-string pkg)) toolchain-packages)
              ]
              (read-file (scripts-dir + "/toolchain-setup.sh"));
        };

        # Modular service for nativelink (nimi pattern)
        mk-nativelink-service =
          { script }:
          _: _: {
            _class = "service";
            config.process.argv = [ "${script}/bin/${script.name}" ];
          };

        # ──────────────────────────────────────────────────────────────────────
        # Nix Proxy: mitmproxy-based caching proxy for build-time fetches
        # ──────────────────────────────────────────────────────────────────────

        # Python addon for mitmproxy - caches responses by content hash

        # Allowlist as comma-separated string

        # Nix proxy entrypoint script

        # ──────────────────────────────────────────────────────────────────────
        # Fly.io configuration (generated from options)
        # ──────────────────────────────────────────────────────────────────────

        scheduler-fly-toml = write-text "scheduler.toml" (
          replace-strings
            [ "@appPrefix@" "@region@" "@schedulerPort@" "@schedulerMemory@" "@schedulerCpus@" ]
            [
              cfg.fly.app-prefix
              cfg.fly.region
              (to-string cfg.scheduler.port)
              cfg.scheduler.memory
              (to-string cfg.scheduler.cpus)
            ]
            (read-file (scripts-dir + "/scheduler-fly.toml"))
        );

        cas-fly-toml = write-text "cas.toml" (
          replace-strings
            [ "@appPrefix@" "@region@" "@casPort@" "@casVolumeSize@" "@casMemory@" "@casCpus@" ]
            [
              cfg.fly.app-prefix
              cfg.fly.region
              (to-string cfg.cas.port)
              cfg.cas.volume-size
              cfg.cas.memory
              (to-string cfg.cas.cpus)
            ]
            (read-file (scripts-dir + "/cas-fly.toml"))
        );

        worker-fly-toml = write-text "worker.toml" (
          replace-strings
            [
              "@appPrefix@"
              "@region@"
              "@workerCount@"
              "@workerCpus@"
              "@totalCores@"
              "@workerVolumeSize@"
              "@workerMemory@"
              "@workerCpuKind@"
            ]
            [
              cfg.fly.app-prefix
              cfg.fly.region
              (to-string cfg.worker.count)
              (to-string cfg.worker.cpus)
              (to-string (cfg.worker.count * cfg.worker.cpus))
              cfg.worker.volume-size
              cfg.worker.memory
              cfg.worker.cpu-kind
            ]
            (read-file (scripts-dir + "/worker-fly.toml"))
        );

        builder-fly-toml = write-text "builder.toml" (
          replace-strings
            [ "@appPrefix@" "@region@" "@builderVolumeSize@" "@builderMemory@" "@builderCpus@" ]
            [
              cfg.fly.app-prefix
              cfg.fly.region
              cfg.builder.volume-size
              cfg.builder.memory
              (to-string cfg.builder.cpus)
            ]
            (read-file (scripts-dir + "/builder-fly.toml"))
        );

        # ──────────────────────────────────────────────────────────────────────
        # Deployment scripts for Fly.io
        # Usage: nix run .#nativelink-deploy
        # ──────────────────────────────────────────────────────────────────────

        # Single unified deploy script - idempotent, does everything
        deploy-all = write-shell-application {
          name = "nativelink-deploy";
          "runtimeInputs" = with pkgs; [
            skopeo
            flyctl
            gh
            coreutils
            jq
          ];
          text =
            replace-strings
              [
                "@appPrefix@"
                "@region@"
                "@workerCount@"
                "@workerCpus@"
                "@totalCores@"
                "@casVolumeSize@"
                "@workerVolumeSize@"
                "@builderVolumeSize@"
                "@schedulerFlyToml@"
                "@casFlyToml@"
                "@workerFlyToml@"
                "@builderFlyToml@"
              ]
              [
                cfg.fly.app-prefix
                cfg.fly.region
                (to-string cfg.worker.count)
                (to-string cfg.worker.cpus)
                (to-string (cfg.worker.count * cfg.worker.cpus))
                cfg.cas.volume-size
                cfg.worker.volume-size
                cfg.builder.volume-size
                (to-string scheduler-fly-toml)
                (to-string cas-fly-toml)
                (to-string worker-fly-toml)
                (to-string builder-fly-toml)
              ]
              (read-file (scripts-dir + "/deploy-all.sh"));
        };

        deploy-scheduler = write-shell-application {
          name = "nativelink-deploy-scheduler";
          "runtimeInputs" = with pkgs; [ flyctl ];
          text = ''
            exec ${deploy-all}/bin/nativelink-deploy
          '';
        };

        deploy-cas = write-shell-application {
          name = "nativelink-deploy-cas";
          "runtimeInputs" = with pkgs; [ flyctl ];
          text = ''
            exec ${deploy-all}/bin/nativelink-deploy
          '';
        };

        deploy-worker = write-shell-application {
          name = "nativelink-deploy-worker";
          "runtimeInputs" = with pkgs; [ flyctl ];
          text = ''
            exec ${deploy-all}/bin/nativelink-deploy
          '';
        };

        # Status check script
        status-script = write-shell-application {
          name = "nativelink-status";
          "runtimeInputs" = with pkgs; [ flyctl ];
          text = replace-strings [ "@appPrefix@" ] [ cfg.fly.app-prefix ] (
            read-file (scripts-dir + "/status.sh")
          );
        };

        # Logs script
        logs-script = write-shell-application {
          name = "nativelink-logs";
          "runtimeInputs" = with pkgs; [ flyctl ];
          text = replace-strings [ "@appPrefix@" ] [ cfg.fly.app-prefix ] (
            read-file (scripts-dir + "/logs.sh")
          );
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

          # Worker container (minimal - toolchain fetched at runtime)
          # Tools live on the volume at /data/nix, fetched from cache.nixos.org
          # This keeps the image under Fly's 8GB limit
          nativelink-worker = {
            "systemPackages" = [
              nativelink
              worker-script
              worker-setup-script
              pkgs.nix
              pkgs.coreutils
              pkgs.bash
              pkgs.cacert
            ];

            services.worker = {
              imports = [ (mk-nativelink-service { script = worker-script; } { inherit lib pkgs; }) ];
            };

            registries = [ cfg.registry ];

            "extraEnv" = {
              RUST_LOG = "info";
              NIX_SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            };
          };

          # Builder container - nix + git + skopeo for remote builds
          nativelink-builder = {
            "systemPackages" = with pkgs; [
              nix
              git
              skopeo
              openssh
              coreutils
              bash
              gnugrep
              gnutar
              gzip
              curl
              jq
              cacert
            ];

            registries = [ cfg.registry ];

            "extraEnv" = {
              NIX_SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            };
          };
        };

        # ────────────────────────────────────────────────────────────────────
        # Export configs and scripts as packages for inspection/debugging
        # ────────────────────────────────────────────────────────────────────

        packages = {
          # NativeLink JSON configs (for debugging)
          nativelink-scheduler-config = scheduler-config;
          nativelink-cas-config = cas-config;
          nativelink-worker-config = worker-config;

          # Fly.io TOML configs (generated from options)
          nativelink-scheduler-fly-toml = scheduler-fly-toml;
          nativelink-cas-fly-toml = cas-fly-toml;
          nativelink-worker-fly-toml = worker-fly-toml;
          nativelink-builder-fly-toml = builder-fly-toml;

          # Entrypoint scripts (used by containers)
          nativelink-scheduler-script = scheduler-script;
          nativelink-cas-script = cas-script;
          nativelink-worker-script = worker-script;
          nativelink-worker-setup = worker-setup-script;
          nativelink-toolchain-manifest = toolchain-manifest;

          # THE deploy script - does everything
          # nix run .#nativelink-deploy
          nativelink-deploy = deploy-all;

          # Aliases for convenience
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
