# nix/modules/flake/container/default.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                       // container, namespace, and vm isolation //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "Case felt the edge, that edge that came when you dealt with the
#      Yakuza's puppet theater."
#
#                                                         — Neuromancer
#
# Provides runnable tools via `nix run`:
#   - crane-inspect, crane-pull (OCI image operations)
#   - unshare-run, unshare-gpu (bwrap namespace runners)
#   - isospin-run, isospin-build (Isospin/Firecracker VM)
#   - cloud-hypervisor-run, cloud-hypervisor-gpu (Cloud Hypervisor + VFIO)
#   - fhs-run, gpu-run (simple namespace runners)
#   - vfio-bind, vfio-unbind, vfio-list
#
# Philosophy:
#   - Namespaces, not daemons (bwrap, not Docker)
#   - Presentation, not mutation (bind mounts, not patchelf)
#   - VM isolation for network builds (Isospin, not sandbox escape)
#   - VFIO for GPU passthrough (cloud-hypervisor-gpu, not nvidia-docker)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib, ... }:
let
  # ────────────────────────────────────────────────────────────────────────────
  # // lib aliases (lisp-case) //
  # ────────────────────────────────────────────────────────────────────────────

  mk-enable-option = lib.mkEnableOption;
  mk-option = lib.mkOption;
  mk-if = lib.mkIf;
  make-bin-path = lib.makeBinPath;
  optional-attrs = lib.optionalAttrs;

  # ────────────────────────────────────────────────────────────────────────────
  # // imports //
  # ────────────────────────────────────────────────────────────────────────────

  kernels = import ./kernels.nix { inherit lib; };
  init-scripts = import ./init-scripts.nix;

  inherit (kernels) fc-kernel ch-kernel;

  # Import nimi-init module (needs pkgs and nimi, so done in perSystem)

  # ────────────────────────────────────────────────────────────────────────────
  # // container module //
  # ────────────────────────────────────────────────────────────────────────────

  container-module =
    { config, lib, ... }:
    let
      cfg = config.aleph.container;
    in
    {
      _class = "flake";

      # ──────────────────────────────────────────────────────────────────────────
      # // options //
      # ──────────────────────────────────────────────────────────────────────────

      options.aleph.container = {
        enable = mk-enable-option "container and VM isolation tools" // {
          default = true;
        };

        isospin = {
          enable = mk-enable-option "Isospin (Firecracker fork) VM tools" // {
            default = true;
          };

          cpus = mk-option {
            type = lib.types.int;
            default = 4;
            description = "Default vCPU count";
          };

          mem-mib = mk-option {
            type = lib.types.int;
            default = 4096;
            description = "Default memory in MiB";
          };
        };

        cloud-hypervisor = {
          enable = mk-enable-option "Cloud Hypervisor VM tools (VFIO GPU)" // {
            default = true;
          };

          cpus = mk-option {
            type = lib.types.int;
            default = 8;
            description = "Default vCPU count";
          };

          mem-gib = mk-option {
            type = lib.types.int;
            default = 16;
            description = "Default memory in GiB";
          };

          hugepages = mk-option {
            type = lib.types.bool;
            default = false;
            description = "Use hugepages (recommended for GPU, requires pre-allocated hugepages)";
          };
        };
      };

      # ──────────────────────────────────────────────────────────────────────────
      # // config //
      # ──────────────────────────────────────────────────────────────────────────

      config = mk-if cfg.enable {
        perSystem =
          { pkgs, system, ... }:
          let
            # ──────────────────────────────────────────────────────────────────────
            # // kernel packages //
            # ──────────────────────────────────────────────────────────────────────

            fc-kernel-pkg =
              if fc-kernel ? ${system} then
                pkgs.fetchurl {
                  inherit (fc-kernel.${system}) url hash;
                }
              else
                null;

            ch-kernel-pkg =
              if ch-kernel ? ${system} then
                pkgs.fetchurl {
                  inherit (ch-kernel.${system}) url hash;
                }
              else
                null;

            # Map Nix system to OCI platform

            # ──────────────────────────────────────────────────────────────────────
            # // init script files //
            # ──────────────────────────────────────────────────────────────────────

            isospin-run-init = pkgs.writeText "isospin-run-init" init-scripts.fc-run-init;
            isospin-build-init = pkgs.writeText "isospin-build-init" init-scripts.fc-build-init;
            cloud-hypervisor-run-init = pkgs.writeText "cloud-hypervisor-run-init" init-scripts.ch-run-init;
            cloud-hypervisor-gpu-init = pkgs.writeText "cloud-hypervisor-gpu-init" init-scripts.ch-gpu-init;

            # ──────────────────────────────────────────────────────────────────────
            # // jq filters //
            # ──────────────────────────────────────────────────────────────────────

            # ────────────────────────────────────────────────────────────────────────
            # // compiled haskell scripts //
            # ──────────────────────────────────────────────────────────────────────
            #
            # These are compiled from nix/scripts/*.hs via the aleph.script overlay.
            # Type-safe, fast startup (~2ms), no bash variable injection risks.

            inherit (pkgs.aleph.script) compiled;

            # ──────────────────────────────────────────────────────────────────────
            # // dhall config for firecracker //
            # ──────────────────────────────────────────────────────────────────────
            #
            # Generates a typed Dhall config with real Nix store paths.
            # The isospin-run binary reads this at startup via CONFIG_FILE env var.

            isospin-dhall-config = pkgs.writeText "isospin-config.dhall" ''
              { kernel = "${fc-kernel-pkg}"
              , busybox = "${pkgs.pkgsStatic.busybox}/bin/busybox"
              , initScript = "${isospin-run-init}"
              , buildInitScript = "${isospin-build-init}"
              , defaultCpus = ${toString cfg.isospin.cpus}
              , defaultMemMib = ${toString cfg.isospin.mem-mib}
              , cacheDir = "/var/cache/aleph/oci"
              }
            '';

            # Wrapped isospin-run with Dhall config injected
            isospin-run-wrapped = mk-if (fc-kernel-pkg != null) (
              pkgs.runCommand "isospin-run" { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
                mkdir -p $out/bin
                makeWrapper ${compiled.isospin-run}/bin/isospin-run $out/bin/isospin-run \
                  --set CONFIG_FILE ${isospin-dhall-config} \
                  --prefix PATH : ${
                    make-bin-path [
                      pkgs.crane
                      pkgs.gnutar
                      pkgs.fakeroot
                      pkgs.genext2fs
                      pkgs.e2fsprogs
                      pkgs.firecracker # TODO: replace with isospin package
                      pkgs.iproute2 # for ip command (TAP setup)
                      pkgs.iptables # for NAT masquerading
                    ]
                  }
              ''
            );

            # ──────────────────────────────────────────────────────────────────────
            # // dhall config for cloud hypervisor //
            # ──────────────────────────────────────────────────────────────────────
            #
            # Generates a typed Dhall config with real Nix store paths.
            # The cloud-hypervisor-run/cloud-hypervisor-gpu binaries read this at startup via CONFIG_FILE env var.

            cloud-hypervisor-dhall-config = pkgs.writeText "cloud-hypervisor-config.dhall" ''
              { chKernel = "${ch-kernel-pkg}"
              , chBusybox = "${pkgs.pkgsStatic.busybox}/bin/busybox"
              , chInitScript = "${cloud-hypervisor-run-init}"
              , chGpuInitScript = Some "${cloud-hypervisor-gpu-init}"
              , chDefaultCpus = ${toString cfg.cloud-hypervisor.cpus}
              , chDefaultMemMib = ${toString (cfg.cloud-hypervisor.mem-gib * 1024)}
              , chHugepages = ${if cfg.cloud-hypervisor.hugepages then "True" else "False"}
              , chCacheDir = "/var/cache/aleph/oci"
              }
            '';

            # Wrapped cloud-hypervisor-run with Dhall config injected
            cloud-hypervisor-run-wrapped = mk-if (ch-kernel-pkg != null) (
              pkgs.runCommand "cloud-hypervisor-run" { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
                mkdir -p $out/bin
                makeWrapper ${compiled.cloud-hypervisor-run}/bin/cloud-hypervisor-run $out/bin/cloud-hypervisor-run \
                  --set CONFIG_FILE ${cloud-hypervisor-dhall-config} \
                  --prefix PATH : ${
                    make-bin-path [
                      pkgs.crane
                      pkgs.gnutar
                      pkgs.fakeroot
                      pkgs.genext2fs
                      pkgs.e2fsprogs
                      pkgs.cloud-hypervisor
                    ]
                  }
              ''
            );

            # Wrapped cloud-hypervisor-gpu with Dhall config injected
            cloud-hypervisor-gpu-wrapped = mk-if (ch-kernel-pkg != null) (
              pkgs.runCommand "cloud-hypervisor-gpu" { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
                mkdir -p $out/bin
                makeWrapper ${compiled.cloud-hypervisor-gpu}/bin/cloud-hypervisor-gpu $out/bin/cloud-hypervisor-gpu \
                  --set CONFIG_FILE ${cloud-hypervisor-dhall-config} \
                  --prefix PATH : ${
                    make-bin-path [
                      pkgs.crane
                      pkgs.gnutar
                      pkgs.fakeroot
                      pkgs.genext2fs
                      pkgs.e2fsprogs
                      pkgs.cloud-hypervisor
                      pkgs.pciutils
                      compiled.vfio-bind
                      compiled.vfio-unbind
                    ]
                  }
              ''
            );

            # ──────────────────────────────────────────────────────────────────────
            # // armitage builder (OCI container via nix2gpu) //
            # ──────────────────────────────────────────────────────────────────────
            #
            # OCI container with Nimi + Armitage + Nix for witnessed builds.
            # All network fetches go through Armitage proxy (TLS MITM).
            # Attestation log written to /var/log/armitage/fetches.jsonl
            #
            # Deploy to: Fly.io, Docker, Podman, Kubernetes
            # Build: nix build .#armitage-builder
            # Run:   docker load < result && docker run -it armitage-builder

            # Armitage startup script for nimi service
            # Environment variables are embedded since nimi doesn't support process.environment
            armitage-startup-script = pkgs.writeShellApplication {
              name = "armitage-startup";
              runtimeInputs = with pkgs; [
                coreutils
                procps
              ];
              text = ''
                # Create directories for armitage
                mkdir -p /var/cache/armitage /var/log/armitage /etc/ssl/armitage

                # Set environment for armitage proxy
                export PROXY_PORT="8888"
                export PROXY_CACHE_DIR="/var/cache/armitage"
                export PROXY_LOG_DIR="/var/log/armitage"
                export PROXY_CERT_DIR="/etc/ssl/armitage"

                # Start Armitage proxy
                echo ":: Starting Armitage proxy on :8888..."
                exec ${pkgs.armitage-proxy}/bin/armitage-proxy
              '';
            };

            # Witnessed build wrapper - runs command with attestation collection
            # Usage: witnessed-build <output-dir> -- <command> [args...]
            # After command completes, copies attestation log to <output-dir>/.attestations.jsonl
            witnessed-build = pkgs.writeShellApplication {
              name = "witnessed-build";
              runtimeInputs = with pkgs; [
                coreutils
                jq
              ];
              text = ''
                OUTPUT_DIR="$1"
                shift
                if [ "$1" = "--" ]; then shift; fi

                # Clear attestation log before build
                ATTESTATION_LOG="/var/log/armitage/fetches.jsonl"
                : > "$ATTESTATION_LOG"

                # Run the actual command
                EXIT_CODE=0
                "$@" || EXIT_CODE=$?

                # Copy attestations to output directory
                if [ -f "$ATTESTATION_LOG" ] && [ -s "$ATTESTATION_LOG" ]; then
                  cp "$ATTESTATION_LOG" "$OUTPUT_DIR/.attestations.jsonl"
                  FETCH_COUNT=$(wc -l < "$ATTESTATION_LOG")
                  echo ":: Witnessed $FETCH_COUNT network fetch(es)"
                fi

                exit $EXIT_CODE
              '';
            };

            # Armitage service module for nimi (follows nativelink pattern)
            mk-armitage-service = _: _: {
              _class = "service";
              config.process.argv = [ "${armitage-startup-script}/bin/armitage-startup" ];
            };

          in
          {
            packages = optional-attrs (system == "x86_64-linux" || system == "aarch64-linux") {

              # ──────────────────────────────────────────────────────────────────
              # // crane tools (OCI image operations) //
              # ──────────────────────────────────────────────────────────────────

              inherit (compiled)
                crane-inspect
                crane-pull
                ;

              # ──────────────────────────────────────────────────────────────────
              # // unshare tools (bwrap namespace runners) //
              # ──────────────────────────────────────────────────────────────────

              inherit (compiled)
                unshare-run
                unshare-gpu
                ;

              # ──────────────────────────────────────────────────────────────────
              # // isospin tools (Firecracker fork) //
              # ──────────────────────────────────────────────────────────────────

              # Compiled Haskell version (type-safe, ~2ms startup)
              # Uses Dhall config for store path injection
              isospin-run = mk-if (cfg.isospin.enable && fc-kernel-pkg != null) isospin-run-wrapped;

              # ──────────────────────────────────────────────────────────────────
              # // cloud hypervisor tools //
              # ──────────────────────────────────────────────────────────────────

              # Compiled Haskell version (type-safe, ~2ms startup)
              # Uses Dhall config for store path injection
              cloud-hypervisor-run = mk-if (
                cfg.cloud-hypervisor.enable && ch-kernel-pkg != null
              ) cloud-hypervisor-run-wrapped;
              cloud-hypervisor-gpu = mk-if (
                cfg.cloud-hypervisor.enable && ch-kernel-pkg != null
              ) cloud-hypervisor-gpu-wrapped;

              # ──────────────────────────────────────────────────────────────────
              # // vfio helpers //
              # ──────────────────────────────────────────────────────────────────
              #
              # These are compiled Haskell scripts from aleph.script.compiled.
              # Type-safe, ~2ms startup, no bash injection risks.

              inherit (compiled) vfio-bind vfio-unbind vfio-list;

              # ──────────────────────────────────────────────────────────────────
              # // namespace runners //
              # ──────────────────────────────────────────────────────────────────
              #
              # Compiled Haskell scripts for FHS/GPU namespace runners.

              inherit (compiled) fhs-run gpu-run;

              # ──────────────────────────────────────────────────────────────────
              # // armitage builder //
              # ──────────────────────────────────────────────────────────────────
              #
              # OCI container with Nimi + Armitage for witnessed builds.
              # Built via nix2gpu module - see nix2gpu.armitage-builder below.
              # Usage: nix build .#armitage-builder
            };

            # ──────────────────────────────────────────────────────────────────────
            # // nix2gpu container definitions //
            # ──────────────────────────────────────────────────────────────────────
            #
            # OCI containers built via nix2gpu with Ubuntu 24.04 base.
            # These containers work on Fly.io, Docker, Podman, Kubernetes.
            #
            # Build:   nix build .#armitage-builder
            # Load:    docker load < result
            # Run:     docker run -it armitage-builder
            # Push:    nix run .#armitage-builder.copyToGithub

            nix2gpu = {
              # Armitage Builder - Witnessed build environment
              # All network fetches go through Armitage TLS MITM proxy.
              # Attestation log written to /var/log/armitage/fetches.jsonl
              armitage-builder = {
                "systemPackages" = with pkgs; [
                  # Core build tools
                  nix
                  git
                  coreutils
                  bash
                  gnugrep
                  gnutar
                  gzip
                  curl
                  jq
                  cacert

                  # Armitage proxy and build wrapper
                  armitage-proxy
                  armitage-startup-script
                  witnessed-build

                  # Process management
                  procps
                ];

                # Nimi service for armitage proxy
                services.armitage = {
                  imports = [ (mk-armitage-service { inherit lib pkgs; }) ];
                };

                "extraEnv" = {
                  # Proxy environment for all processes (uppercase only for nix2gpu)
                  HTTP_PROXY = "http://127.0.0.1:8888";
                  HTTPS_PROXY = "http://127.0.0.1:8888";
                  # Note: SSL_CERT_FILE set at runtime after CA is generated
                  NIX_SSL_CERT_FILE = "/etc/ssl/armitage/ca.pem";
                  SSL_CERT_FILE = "/etc/ssl/armitage/ca.pem";
                };
              };
            };
          };
      };
    };

in
container-module
