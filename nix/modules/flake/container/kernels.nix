# nix/modules/flake/container/kernels.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // vm kernel configs //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "The matrix had a vastness that still frightened him, even now."
#
#                                                         — Neuromancer
#
# VM kernel configurations for Firecracker and Cloud Hypervisor.
# Type-checked via lib.evalModules pattern.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
let
  # ────────────────────────────────────────────────────────────────────────────
  # // type definitions //
  # ────────────────────────────────────────────────────────────────────────────

  # Kernel source definition for a specific architecture
  kernel-source-type = lib.types.submodule {
    options = {
      url = lib.mkOption {
        type = lib.types.str;
        description = "URL to fetch the kernel from";
      };
      hash = lib.mkOption {
        type = lib.types.str;
        description = "SRI hash of the kernel";
      };
    };
  };

  # Full kernel definition with version and per-arch sources
  kernel-type = lib.types.submodule {
    options = {
      version = lib.mkOption {
        type = lib.types.str;
        default = "unknown";
        description = "Kernel version string";
      };
      x86_64-linux = lib.mkOption {
        type = lib.types.nullOr kernel-source-type;
        default = null;
        description = "x86_64-linux kernel source";
      };
      aarch64-linux = lib.mkOption {
        type = lib.types.nullOr kernel-source-type;
        default = null;
        description = "aarch64-linux kernel source";
      };
    };
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // kernel configurations //
  # ────────────────────────────────────────────────────────────────────────────

  kernels-module = {
    options = {
      fc-kernel = lib.mkOption {
        type = kernel-type;
        description = "Firecracker kernel configuration";
      };
      ch-kernel = lib.mkOption {
        type = kernel-type;
        description = "Cloud Hypervisor kernel configuration";
      };
      busybox-cmds = lib.mkOption {
        type = lib.types.listOf lib.types.str;
        description = "Busybox commands to inject into VMs";
      };
    };

    config = {
      # AWS Firecracker kernel - tested, virtio baked in
      fc-kernel = {
        version = "5.10.225";
        x86_64-linux = {
          url = "https://s3.amazonaws.com/spec.ccfc.min/firecracker-ci/v1.11/x86_64/vmlinux-5.10.225";
          hash = "sha256-I7MEfffa2jUA0GyAEswDC5IdoB4hNzX3cX0hZs/PXwY=";
        };
        aarch64-linux = {
          url = "https://s3.amazonaws.com/spec.ccfc.min/firecracker-ci/v1.11/aarch64/vmlinux-5.10.225";
          hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="; # TODO
        };
      };

      # Cloud Hypervisor kernel
      ch-kernel = {
        version = "6.2";
        x86_64-linux = {
          url = "https://github.com/cloud-hypervisor/linux/releases/download/ch-release-v6.2-20240908/vmlinux";
          hash = "sha256-tdca3rES7uzDB4b8b+7fA2f0xukAIiTN1gBbeGw101c=";
        };
      };

      # Busybox commands to inject into VMs
      busybox-cmds = [
        "sh"
        "mount"
        "hostname"
        "ip"
        "free"
        "clear"
        "setsid"
        "cttyhack"
        "nproc"
        "awk"
        "cat"
        "echo"
        "chmod"
        "mkdir"
        "ln"
        "rm"
        "cp"
        "mv"
        "ls"
        "grep"
        "sed"
        "tar"
        "poweroff"
        "reboot"
        "dmesg"
        "sleep"
      ];
    };
  };

  # Evaluate the module to get type-checked config
  evaluated = lib.evalModules {
    modules = [ kernels-module ];
  };

in
{
  _class = "flake";
}
// evaluated.config
