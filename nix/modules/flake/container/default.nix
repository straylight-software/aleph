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
#   - oci-run, oci-inspect, oci-pull, oci-gpu
#   - fc-run, fc-build (Firecracker)
#   - ch-run, ch-gpu (Cloud Hypervisor + VFIO)
#   - fhs-run, gpu-run (namespace runners)
#   - vfio-bind, vfio-unbind, vfio-list
#
# Philosophy:
#   - Namespaces, not daemons (bwrap, not Docker)
#   - Presentation, not mutation (bind mounts, not patchelf)
#   - VM isolation for network builds (Firecracker, not sandbox escape)
#   - VFIO for GPU passthrough (ch-gpu, not nvidia-docker)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib, ... }:
let
  # ────────────────────────────────────────────────────────────────────────────
  # // imports //
  # ────────────────────────────────────────────────────────────────────────────

  kernels = import ./kernels.nix { inherit lib; };
  init-scripts = import ./init-scripts.nix;

  inherit (kernels) fc-kernel ch-kernel;

  # ────────────────────────────────────────────────────────────────────────────
  # // container module //
  # ────────────────────────────────────────────────────────────────────────────

  container-module =
    { config, lib, ... }:
    let
      cfg = config.aleph-naught.container;
    in
    {
      _class = "flake";

      # ──────────────────────────────────────────────────────────────────────────
      # // options //
      # ──────────────────────────────────────────────────────────────────────────

      options.aleph-naught.container = {
        enable = lib.mkEnableOption "container and VM isolation tools" // {
          default = true;
        };

        firecracker = {
          enable = lib.mkEnableOption "Firecracker VM tools" // {
            default = true;
          };

          cpus = lib.mkOption {
            type = lib.types.int;
            default = 4;
            description = "Default vCPU count";
          };

          mem-mib = lib.mkOption {
            type = lib.types.int;
            default = 4096;
            description = "Default memory in MiB";
          };
        };

        cloud-hypervisor = {
          enable = lib.mkEnableOption "Cloud Hypervisor VM tools (VFIO GPU)" // {
            default = true;
          };

          cpus = lib.mkOption {
            type = lib.types.int;
            default = 8;
            description = "Default vCPU count";
          };

          mem-gib = lib.mkOption {
            type = lib.types.int;
            default = 16;
            description = "Default memory in GiB";
          };

          hugepages = lib.mkOption {
            type = lib.types.bool;
            default = false;
            description = "Use hugepages (recommended for GPU, requires pre-allocated hugepages)";
          };
        };
      };

      # ──────────────────────────────────────────────────────────────────────────
      # // config //
      # ──────────────────────────────────────────────────────────────────────────

      config = lib.mkIf cfg.enable {
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

            fc-run-init = pkgs.writeText "fc-run-init" init-scripts.fc-run-init;
            fc-build-init = pkgs.writeText "fc-build-init" init-scripts.fc-build-init;
            ch-run-init = pkgs.writeText "ch-run-init" init-scripts.ch-run-init;
            ch-gpu-init = pkgs.writeText "ch-gpu-init" init-scripts.ch-gpu-init;

            # ──────────────────────────────────────────────────────────────────────
            # // jq filters //
            # ──────────────────────────────────────────────────────────────────────

            # ────────────────────────────────────────────────────────────────────────
            # // compiled haskell scripts //
            # ──────────────────────────────────────────────────────────────────────
            #
            # These are compiled from nix/scripts/*.hs via the straylight.script overlay.
            # Type-safe, fast startup (~2ms), no bash variable injection risks.

            inherit (pkgs.straylight.script) compiled;

            # ──────────────────────────────────────────────────────────────────────
            # // dhall config for firecracker //
            # ──────────────────────────────────────────────────────────────────────
            #
            # Generates a typed Dhall config with real Nix store paths.
            # The fc-run binary reads this at startup via CONFIG_FILE env var.

            fc-dhall-config = pkgs.writeText "fc-config.dhall" ''
              { kernel = "${fc-kernel-pkg}"
              , busybox = "${pkgs.pkgsStatic.busybox}/bin/busybox"
              , initScript = "${fc-run-init}"
              , buildInitScript = "${fc-build-init}"
              , defaultCpus = ${toString cfg.firecracker.cpus}
              , defaultMemMib = ${toString cfg.firecracker.mem-mib}
              , cacheDir = "/var/cache/straylight/oci"
              }
            '';

            # Wrapped fc-run with Dhall config injected
            fc-run-wrapped = lib.mkIf (fc-kernel-pkg != null) (
              pkgs.runCommand "fc-run" { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
                mkdir -p $out/bin
                makeWrapper ${compiled.fc-run}/bin/fc-run $out/bin/fc-run \
                  --set CONFIG_FILE ${fc-dhall-config} \
                  --prefix PATH : ${
                    lib.makeBinPath [
                      pkgs.crane
                      pkgs.gnutar
                      pkgs.fakeroot
                      pkgs.genext2fs
                      pkgs.e2fsprogs
                      pkgs.firecracker
                    ]
                  }
              ''
            );

            # ──────────────────────────────────────────────────────────────────────
            # // dhall config for cloud hypervisor //
            # ──────────────────────────────────────────────────────────────────────
            #
            # Generates a typed Dhall config with real Nix store paths.
            # The ch-run/ch-gpu binaries read this at startup via CONFIG_FILE env var.

            ch-dhall-config = pkgs.writeText "ch-config.dhall" ''
              { chKernel = "${ch-kernel-pkg}"
              , chBusybox = "${pkgs.pkgsStatic.busybox}/bin/busybox"
              , chInitScript = "${ch-run-init}"
              , chGpuInitScript = Some "${ch-gpu-init}"
              , chDefaultCpus = ${toString cfg.cloud-hypervisor.cpus}
              , chDefaultMemMib = ${toString (cfg.cloud-hypervisor.mem-gib * 1024)}
              , chHugepages = ${if cfg.cloud-hypervisor.hugepages then "True" else "False"}
              , chCacheDir = "/var/cache/straylight/oci"
              }
            '';

            # Wrapped ch-run with Dhall config injected
            ch-run-wrapped = lib.mkIf (ch-kernel-pkg != null) (
              pkgs.runCommand "ch-run" { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
                mkdir -p $out/bin
                makeWrapper ${compiled.ch-run}/bin/ch-run $out/bin/ch-run \
                  --set CONFIG_FILE ${ch-dhall-config} \
                  --prefix PATH : ${
                    lib.makeBinPath [
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

            # Wrapped ch-gpu with Dhall config injected
            ch-gpu-wrapped = lib.mkIf (ch-kernel-pkg != null) (
              pkgs.runCommand "ch-gpu" { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
                mkdir -p $out/bin
                makeWrapper ${compiled.ch-gpu}/bin/ch-gpu $out/bin/ch-gpu \
                  --set CONFIG_FILE ${ch-dhall-config} \
                  --prefix PATH : ${
                    lib.makeBinPath [
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
            # // busybox injection //
            # ──────────────────────────────────────────────────────────────────────

          in
          {
            packages = lib.optionalAttrs (system == "x86_64-linux" || system == "aarch64-linux") {

              # ──────────────────────────────────────────────────────────────────
              # // oci tools //
              # ──────────────────────────────────────────────────────────────────
              #
              # Compiled Haskell scripts for OCI container operations.
              # Type-safe, ~2ms startup, cached image handling, GPU passthrough.

              inherit (compiled)
                oci-run
                oci-gpu
                oci-inspect
                oci-pull
                ;

              # ──────────────────────────────────────────────────────────────────
              # // firecracker tools //
              # ──────────────────────────────────────────────────────────────────

              # Compiled Haskell version (type-safe, ~2ms startup)
              # Uses Dhall config for store path injection
              fc-run = lib.mkIf (cfg.firecracker.enable && fc-kernel-pkg != null) fc-run-wrapped;

              # ──────────────────────────────────────────────────────────────────
              # // cloud hypervisor tools //
              # ──────────────────────────────────────────────────────────────────

              # Compiled Haskell version (type-safe, ~2ms startup)
              # Uses Dhall config for store path injection
              ch-run = lib.mkIf (cfg.cloud-hypervisor.enable && ch-kernel-pkg != null) ch-run-wrapped;
              ch-gpu = lib.mkIf (cfg.cloud-hypervisor.enable && ch-kernel-pkg != null) ch-gpu-wrapped;

              # ──────────────────────────────────────────────────────────────────
              # // vfio helpers //
              # ──────────────────────────────────────────────────────────────────
              #
              # These are compiled Haskell scripts from straylight.script.compiled.
              # Type-safe, ~2ms startup, no bash injection risks.

              inherit (compiled) vfio-bind vfio-unbind vfio-list;

              # ──────────────────────────────────────────────────────────────────
              # // namespace runners //
              # ──────────────────────────────────────────────────────────────────
              #
              # Compiled Haskell scripts for FHS/GPU namespace runners.

              inherit (compiled) fhs-run gpu-run;
            };
          };
      };
    };

in
container-module
