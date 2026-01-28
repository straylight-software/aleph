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
  nimi,
}:
let

  # ────────────────────────────────────────────────────────────────────────────
  # // lisp-case aliases //
  # ────────────────────────────────────────────────────────────────────────────
  read-file = builtins.readFile;
  replace-strings = builtins.replaceStrings;

  get-exe = lib.getExe;
  map-attrs' = lib.mapAttrs';
  name-value-pair = lib.nameValuePair;
  inherit (lib) optional;
  optional-string = lib.optionalString;
  to-upper = lib.toUpper;

  # ────────────────────────────────────────────────────────────────────────────
  # // vm init script builder //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Creates a single init script that:
  #   1. Mounts virtual filesystems
  #   2. Configures network
  #   3. Optionally waits for GPU
  #   4. Execs into final process (shell or build runner)

  # Script fragments loaded from external files to comply with ALEPH-W003
  network-setup-script = read-file ../scripts/vm-init-network.bash;
  gpu-setup-script = read-file ../scripts/vm-init-gpu.bash;

  # Render Dhall template with env vars (converts attr names to UPPER_SNAKE_CASE)
  render-dhall =
    name: src: vars:
    let
      env-vars = map-attrs' (
        k: v: name-value-pair (to-upper (replace-strings [ "-" ] [ "_" ] k)) (toString v)
      ) vars;
    in
    pkgs.runCommand name
      (
        {
          nativeBuildInputs = [ pkgs.haskellPackages.dhall ];
        }
        // env-vars
      )
      ''
        dhall text --file ${src} > $out
      '';

  mk-vm-init =
    {
      hostname,
      enable-network ? true,
      wait-for-gpu ? false,
      exec-into, # Final command to exec into
    }:
    let
      # Keys use lisp-case so they convert to UPPER_SNAKE_CASE via render-dhall
      # e.g., "network-setup" -> "NETWORK_SETUP"
      script = render-dhall "vm-init-${hostname}" ../scripts/vm-init.dhall {
        inherit hostname;
        inherit exec-into;
        network-setup = optional-string enable-network network-setup-script;
        gpu-setup = optional-string wait-for-gpu gpu-setup-script;
      };
    in
    pkgs.writeShellApplication {
      name = "vm-init-${hostname}";
      # Exclude SC1091: Not following sourced file (Dhall-generated script)
      excludeShellChecks = [ "SC1091" ];
      runtimeInputs =
        with pkgs;
        [
          coreutils
          util-linux
          iproute2
          ncurses
          busybox # for cttyhack
        ]
        ++ optional wait-for-gpu kmod;
      text = ''
        source ${script}
      '';
    };

  # ────────────────────────────────────────────────────────────────────────────
  # // nimi configurations //
  # ────────────────────────────────────────────────────────────────────────────

  isospin-run-nimi = nimi.mkNimiBin {
    settings.binName = "isospin-run-init";
    settings.startup.runOnStartup = get-exe (mk-vm-init {
      hostname = "isospin";
      exec-into = "exec setsid cttyhack /bin/sh";
    });
  };

  isospin-build-nimi = nimi.mkNimiBin {
    settings.binName = "isospin-build-init";
    settings.startup.runOnStartup = get-exe (mk-vm-init {
      hostname = "builder";
      exec-into = ''
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
    settings.startup.runOnStartup = get-exe (mk-vm-init {
      hostname = "cloud-vm";
      exec-into = "exec setsid cttyhack /bin/sh";
    });
  };

  cloud-hypervisor-gpu-nimi = nimi.mkNimiBin {
    settings.binName = "cloud-hypervisor-gpu-init";
    settings.startup.runOnStartup = get-exe (mk-vm-init {
      hostname = "ch-gpu";
      wait-for-gpu = true;
      exec-into = "exec setsid cttyhack /bin/sh";
    });
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // armitage builder //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Firecracker VM with Armitage proxy for witnessed builds.
  # All fetches go through Armitage (TLS MITM), attestations logged.
  #
  # Usage:
  #   1. Boot VM with armitage-builder-nimi as init
  #   2. Armitage starts as Nimi service on port 8888
  #   3. Build runs with HTTP_PROXY=localhost:8888
  #   4. Attestations written to /var/log/armitage/fetches.jsonl
  #
  # The attestation chain provides cryptographic proof of what was fetched.

  # Armitage proxy binary path - needed for startup script
  armitage-proxy-bin = "${pkgs.armitage-proxy}/bin/armitage-proxy";

  armitage-builder-nimi = nimi.mkNimiBin {
    settings.binName = "armitage-builder-init";

    # Startup script sets up VM environment and starts armitage proxy
    # Uses external script with @ARMITAGE_PROXY_BIN@ substitution
    settings.startup.runOnStartup = get-exe (mk-vm-init {
      hostname = "armitage-builder";
      exec-into = builtins.replaceStrings [ "@ARMITAGE_PROXY_BIN@" ] [ armitage-proxy-bin ] (
        builtins.readFile ./scripts/armitage-builder-exec.sh
      );
    });
  };

  # Armitage builder exec script - uses external file with placeholder substitution
  # For microVM rootfs: uses /usr/local/bin/armitage-proxy (baked into image)
  armitage-exec-into-rootfs =
    builtins.replaceStrings [ "@ARMITAGE_PROXY_BIN@" ] [ "/usr/local/bin/armitage-proxy" ]
      (builtins.readFile ./scripts/armitage-builder-exec.sh);

  # Standalone startup script for armitage-builder (for use in microVM rootfs)
  # This is the raw script content, not wrapped by writeShellApplication
  armitage-builder-startup-script-raw =
    render-dhall "vm-init-armitage-builder-raw" ../scripts/vm-init.dhall
      {
        hostname = "armitage-builder";
        exec-into = armitage-exec-into-rootfs;
        network-setup = network-setup-script;
        gpu-setup = "";
      };

  armitage-builder-startup-script = mk-vm-init {
    hostname = "armitage-builder";
    exec-into = armitage-exec-into-rootfs;
  };

in
{
  inherit
    isospin-run-nimi
    isospin-build-nimi
    cloud-hypervisor-run-nimi
    cloud-hypervisor-gpu-nimi
    armitage-builder-nimi
    armitage-builder-startup-script
    armitage-builder-startup-script-raw
    mk-vm-init
    ;
}
