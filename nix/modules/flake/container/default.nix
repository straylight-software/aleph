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

        isospin = {
          enable = lib.mkEnableOption "Isospin (Firecracker fork) VM tools" // {
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
            # These are compiled from nix/scripts/*.hs via the straylight.script overlay.
            # Type-safe, fast startup (~2ms), no bash variable injection risks.

            inherit (pkgs.straylight.script) compiled;

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
              , cacheDir = "/var/cache/straylight/oci"
              }
            '';

            # Wrapped isospin-run with Dhall config injected
            isospin-run-wrapped = lib.mkIf (fc-kernel-pkg != null) (
              pkgs.runCommand "isospin-run" { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
                mkdir -p $out/bin
                makeWrapper ${compiled.isospin-run}/bin/isospin-run $out/bin/isospin-run \
                  --set CONFIG_FILE ${isospin-dhall-config} \
                  --prefix PATH : ${
                    lib.makeBinPath [
                      pkgs.crane
                      pkgs.gnutar
                      pkgs.fakeroot
                      pkgs.genext2fs
                      pkgs.e2fsprogs
                      pkgs.firecracker # TODO: replace with isospin package
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
              , chCacheDir = "/var/cache/straylight/oci"
              }
            '';

            # Wrapped cloud-hypervisor-run with Dhall config injected
            cloud-hypervisor-run-wrapped = lib.mkIf (ch-kernel-pkg != null) (
              pkgs.runCommand "cloud-hypervisor-run" { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
                mkdir -p $out/bin
                makeWrapper ${compiled.cloud-hypervisor-run}/bin/cloud-hypervisor-run $out/bin/cloud-hypervisor-run \
                  --set CONFIG_FILE ${cloud-hypervisor-dhall-config} \
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

            # Wrapped cloud-hypervisor-gpu with Dhall config injected
            cloud-hypervisor-gpu-wrapped = lib.mkIf (ch-kernel-pkg != null) (
              pkgs.runCommand "cloud-hypervisor-gpu" { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
                mkdir -p $out/bin
                makeWrapper ${compiled.cloud-hypervisor-gpu}/bin/cloud-hypervisor-gpu $out/bin/cloud-hypervisor-gpu \
                  --set CONFIG_FILE ${cloud-hypervisor-dhall-config} \
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
              isospin-run = lib.mkIf (cfg.isospin.enable && fc-kernel-pkg != null) isospin-run-wrapped;

              # ──────────────────────────────────────────────────────────────────
              # // cloud hypervisor tools //
              # ──────────────────────────────────────────────────────────────────

              # Compiled Haskell version (type-safe, ~2ms startup)
              # Uses Dhall config for store path injection
              cloud-hypervisor-run = lib.mkIf (
                cfg.cloud-hypervisor.enable && ch-kernel-pkg != null
              ) cloud-hypervisor-run-wrapped;
              cloud-hypervisor-gpu = lib.mkIf (
                cfg.cloud-hypervisor.enable && ch-kernel-pkg != null
              ) cloud-hypervisor-gpu-wrapped;

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
