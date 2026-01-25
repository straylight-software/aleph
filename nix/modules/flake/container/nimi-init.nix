# nix/modules/flake/container/nimi-init.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // nimi vm init //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "Behind the deck, the other construct flickered in and out of focus."
#
#                                                         — Neuromancer
#
# VM init using Nimi (github:weyl-ai/nimi) as PID 1.
#
# Nimi provides proper PID 1 behavior: signal handling, zombie reaping,
# and clean shutdown. The startup script does all VM setup and then execs
# into the final process (shell or build runner).
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  lib,
  pkgs,
}:
let
  inherit (pkgs) nimi;

  # ────────────────────────────────────────────────────────────────────────────
  # // vm init script builder //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Creates a single init script that:
  #   1. Mounts virtual filesystems
  #   2. Configures network
  #   3. Optionally waits for GPU
  #   4. Execs into final process (shell or build runner)

  # Script fragments loaded from external files to comply with WSN-W003
  networkSetupScript = builtins.readFile ../scripts/vm-init-network.bash;
  gpuSetupScript = builtins.readFile ../scripts/vm-init-gpu.bash;

  # Render Dhall template with env vars (converts attr names to UPPER_SNAKE_CASE)
  renderDhall =
    name: src: vars:
    let
      envVars = lib.mapAttrs' (
        k: v: lib.nameValuePair (lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] k)) (toString v)
      ) vars;
    in
    pkgs.runCommand name
      (
        {
          nativeBuildInputs = [ pkgs.haskellPackages.dhall ];
        }
        // envVars
      )
      ''
        dhall text --file ${src} > $out
      '';

  mkVmInit =
    {
      hostname,
      enableNetwork ? true,
      waitForGpu ? false,
      execInto, # Final command to exec into
    }:
    let
      script = renderDhall "vm-init-${hostname}" ../scripts/vm-init.dhall {
        inherit hostname execInto;
        networkSetup = lib.optionalString enableNetwork networkSetupScript;
        gpuSetup = lib.optionalString waitForGpu gpuSetupScript;
      };
    in
    pkgs.writeShellApplication {
      name = "vm-init-${hostname}";
      runtimeInputs =
        with pkgs;
        [
          coreutils
          util-linux
          iproute2
          ncurses
          busybox # for cttyhack
        ]
        ++ lib.optional waitForGpu kmod;
      text = ''
        source ${script}
      '';
    };

  # ────────────────────────────────────────────────────────────────────────────
  # // nimi configurations //
  # ────────────────────────────────────────────────────────────────────────────

  isospin-run-nimi = nimi.mkNimiBin {
    settings.binName = "isospin-run-init";
    settings.startup.runOnStartup = lib.getExe (mkVmInit {
      hostname = "isospin";
      execInto = ''exec setsid cttyhack /bin/sh'';
    });
  };

  isospin-build-nimi = nimi.mkNimiBin {
    settings.binName = "isospin-build-init";
    settings.startup.runOnStartup = lib.getExe (mkVmInit {
      hostname = "builder";
      execInto = ''
        if [ -f /build-cmd ]; then
          chmod +x /build-cmd
          /build-cmd
          echo ":: Build exit: $?"
          echo o > /proc/sysrq-trigger
        else
          echo ":: No /build-cmd, dropping to shell"
          exec /bin/sh
        fi
      '';
    });
  };

  cloud-hypervisor-run-nimi = nimi.mkNimiBin {
    settings.binName = "cloud-hypervisor-run-init";
    settings.startup.runOnStartup = lib.getExe (mkVmInit {
      hostname = "cloud-vm";
      execInto = ''exec setsid cttyhack /bin/sh'';
    });
  };

  cloud-hypervisor-gpu-nimi = nimi.mkNimiBin {
    settings.binName = "cloud-hypervisor-gpu-init";
    settings.startup.runOnStartup = lib.getExe (mkVmInit {
      hostname = "ch-gpu";
      waitForGpu = true;
      execInto = ''exec setsid cttyhack /bin/sh'';
    });
  };

in
{
  inherit
    isospin-run-nimi
    isospin-build-nimi
    cloud-hypervisor-run-nimi
    cloud-hypervisor-gpu-nimi
    mkVmInit
    ;
}
