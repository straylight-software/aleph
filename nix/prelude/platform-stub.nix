# nix/prelude/platform-stub.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                           // platform-stub //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'The matrix has its roots in primitive arcade games,' said the
#      voice-over, 'in early graphics programs and military
#      experimentation with cranial jacks.'
#
#                                                         — Neuromancer
#
# Create stub binaries that fail loudly with actionable error messages.
# Never silently omit a package — tell the user why and what to do.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   mk-platform-stub      create a stub that explains what's missing
#   mk-feature-stub       create a stub for missing features
#   mk-or-stub            use real package if available, else stub
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  lib,
  pkgs,
}:
rec {
  # ─────────────────────────────────────────────────────────────────────────
  # Platform Stubs
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Create a stub binary that explains platform requirements.

    The stub exits with code 1 and prints:
    - What the command does
    - What platform it requires
    - Current platform
    - Alternative approaches

    # Type

    ```
    mk-platform-stub :: {
      name :: String,
      description :: String,
      requires :: String,
      alternatives :: [String],
    } -> Derivation
    ```

    # Examples

    ```nix
    mk-platform-stub {
      name = "isospin-run";
      description = "Run workloads in Isospin (Firecracker) microVMs";
      requires = "Linux with KVM support";
      alternatives = [ "Use Docker" "Use a Linux VM" ];
    }
    ```
  */
  mk-platform-stub =
    {
      name,
      description ? "Run ${name}",
      requires ? "a different platform",
      alternatives ? [ ],
    }:
    let
      alternative-lines =
        if alternatives == [ ] then
          ""
        else
          ''
            echo "" >&2
            echo "Alternatives:" >&2
            ${lib.concatMapStringsSep "\n" (alt: "echo \"  - ${alt}\" >&2") alternatives}
          '';
    in
    pkgs.writeShellApplication {
      inherit name;
      runtimeInputs = [ ];
      text = ''
        echo "ERROR: '${name}' is not available on this platform." >&2
        echo "" >&2
        echo "  Description: ${description}" >&2
        echo "  Requires:    ${requires}" >&2
        echo "  Current:     ${pkgs.stdenv.hostPlatform.system}" >&2
        ${alternative-lines}
        echo "" >&2
        echo "This is a stub. The real '${name}' cannot run here." >&2
        exit 1
      '';
    };

  /**
    Create a stub for a missing feature (not platform-specific).

    # Type

    ```
    mk-feature-stub :: {
      name :: String,
      feature :: String,
      install-hint :: String,
    } -> Derivation
    ```

    # Examples

    ```nix
    mk-feature-stub {
      name = "wasm-eval";
      feature = "builtins.wasm";
      install-hint = "Install straylight-nix from github:determinate-systems/straylight-nix";
    }
    ```
  */
  mk-feature-stub =
    {
      name,
      feature,
      install-hint ? "",
    }:
    pkgs.writeShellApplication {
      inherit name;
      runtimeInputs = [ ];
      text = ''
        echo "ERROR: '${name}' requires the '${feature}' feature." >&2
        echo "" >&2
        ${lib.optionalString (install-hint != "") ''
          echo "To enable this feature:" >&2
          echo "  ${install-hint}" >&2
          echo "" >&2
        ''}
        echo "This is a stub. The feature is not available in your Nix installation." >&2
        exit 1
      '';
    };

  /**
    Create a stub that explains a package isn't available.

    # Type

    ```
    mk-package-stub :: {
      name :: String,
      package :: String,
      reason :: String,
    } -> Derivation
    ```
  */
  mk-package-stub =
    {
      name,
      package ? name,
      reason ? "not available in nixpkgs for this platform",
    }:
    pkgs.writeShellApplication {
      inherit name;
      runtimeInputs = [ ];
      text = ''
        echo "ERROR: '${name}' is not available." >&2
        echo "" >&2
        echo "  Package: ${package}" >&2
        echo "  Reason:  ${reason}" >&2
        echo "  System:  ${pkgs.stdenv.hostPlatform.system}" >&2
        echo "" >&2
        exit 1
      '';
    };

  # ─────────────────────────────────────────────────────────────────────────
  # Conditional Package Builders
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Use a real package if condition is true, otherwise create a stub.

    # Type

    ```
    mk-or-stub :: Bool -> Derivation -> StubArgs -> Derivation
    ```

    # Examples

    ```nix
    mk-or-stub pkgs.stdenv.isLinux firecrackerPackage {
      name = "firecracker";
      description = "Secure microVM runtime";
      requires = "Linux with KVM";
      alternatives = [ "Use Docker" "Use QEMU" ];
    }
    ```
  */
  mk-or-stub =
    condition: package: stub-args:
    if condition then package else mk-platform-stub stub-args;

  /**
    Use a real package if it exists in pkgs, otherwise create a stub.

    # Type

    ```
    mk-if-available :: String -> StubArgs -> Derivation
    ```

    # Examples

    ```nix
    mk-if-available "firecracker" {
      description = "Secure microVM runtime";
      requires = "Linux with KVM";
    }
    ```
  */
  mk-if-available =
    pkg-name: stub-args: pkgs.${pkg-name} or (mk-platform-stub (stub-args // { name = pkg-name; }));

  # ─────────────────────────────────────────────────────────────────────────
  # Common Stubs
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Pre-defined stubs for common Linux-only tools.
  */
  stubs = {
    firecracker = mk-platform-stub {
      name = "firecracker";
      description = "Run workloads in secure Firecracker microVMs";
      requires = "Linux x86_64/aarch64 with KVM support";
      alternatives = [
        "Use Docker: docker run --rm -it <image>"
        "Use a Linux VM via UTM, Parallels, or VirtualBox"
        "Cross-build: nix build --system x86_64-linux .#<target>"
      ];
    };

    cloud-hypervisor = mk-platform-stub {
      name = "cloud-hypervisor";
      description = "Run workloads in Cloud Hypervisor VMs";
      requires = "Linux x86_64/aarch64 with KVM support";
      alternatives = [
        "Use Docker for container workloads"
        "Use QEMU for full VM emulation"
        "Use a Linux development machine"
      ];
    };

    bubblewrap = mk-platform-stub {
      name = "bwrap";
      description = "Create unprivileged namespace sandboxes";
      requires = "Linux with user namespace support";
      alternatives = [
        "Use Docker for containerized builds"
        "Use nix-sandbox (enabled by default on NixOS)"
        "Run directly without sandboxing (less secure)"
      ];
    };

    vfio-bind = mk-platform-stub {
      name = "vfio-bind";
      description = "Bind PCI devices to VFIO driver for passthrough";
      requires = "Linux with VFIO/IOMMU support";
      alternatives = [
        "GPU passthrough is only supported on Linux"
        "Use cloud instances with attached GPUs"
      ];
    };

    nvidia-smi = mk-platform-stub {
      name = "nvidia-smi";
      description = "Query NVIDIA GPU status and configuration";
      requires = "NVIDIA GPU with proprietary driver";
      alternatives = [
        "Use cloud instances with NVIDIA GPUs (AWS, GCP, Lambda Labs)"
        "Check GPU availability: lspci | grep -i nvidia"
      ];
    };
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Helper for container overlay
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Make a Linux-only package with a helpful stub for other platforms.

    # Type

    ```
    linux-only :: Derivation -> StubArgs -> Derivation
    ```
  */
  linux-only =
    package: stub-args: if pkgs.stdenv.isLinux then package else mk-platform-stub stub-args;

  /**
    Make an x86_64-Linux-only package with a helpful stub.
  */
  x86_64-linux-only =
    package: stub-args:
    if pkgs.stdenv.hostPlatform.system == "x86_64-linux" then
      package
    else
      mk-platform-stub (stub-args // { requires = stub-args.requires or "x86_64-linux"; });
}
