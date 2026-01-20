# nix/overlays/container/default.nix
#
# Container, Namespace, and Firecracker overlay
#
# Provides:
#   - pkgs.straylight.container.mk-namespace-env - Create namespace runners
#   - pkgs.straylight.container.mk-oci-rootfs - Content-addressed OCI extraction
#   - pkgs.straylight.container.mk-firecracker-image - Build Firecracker disk images
#   - pkgs.straylight.container.mk-simple-index - Generate PEP 503 indexes
#   - pkgs.straylight.container.extract - Binary extraction with patchelf
#   - pkgs.straylight.container.oci-run - Run OCI images in namespaces
#   - pkgs.straylight.container.fhs-run - Run binaries with FHS layout
#   - pkgs.straylight.container.gpu-run - Run with GPU access
#
# Philosophy:
#   - Namespaces, not daemons (bwrap, not Docker)
#   - Presentation, not mutation (bind mounts, not patchelf)
#   - VM isolation for network builds (Firecracker, not sandbox escape)
#
# Platform support:
#   - Most features require Linux (namespaces, bwrap, Firecracker)
#   - On non-Linux: helpful stubs explain what's needed
#
final: prev:
let
  inherit (prev) lib;

  straylight-lib = import ../../lib { inherit lib; };

  # Import platform stub helpers
  platform-stub = import ../../prelude/platform-stub.nix {
    inherit lib;
    pkgs = final;
  };

  # ══════════════════════════════════════════════════════════════════════════════
  # IMPORTS (only on supported platforms)
  # ══════════════════════════════════════════════════════════════════════════════

  namespace-mod = lib.optionalAttrs final.stdenv.isLinux (
    import ./namespace.nix { inherit final lib straylight-lib; }
  );
  oci-mod = lib.optionalAttrs final.stdenv.isLinux (import ./oci.nix { inherit final lib; });
  firecracker-mod = lib.optionalAttrs final.stdenv.isLinux (
    import ./firecracker.nix { inherit final lib; }
  );
  extract-mod = import ./extract.nix { inherit final lib straylight-lib; };
  ngc-mod = lib.optionalAttrs final.stdenv.isLinux (
    import ./ngc.nix { inherit final lib straylight-lib; }
  );
  pep503-mod = import ./pep503.nix { inherit final; };

  # ══════════════════════════════════════════════════════════════════════════════
  # FHS/GPU RUNNERS — compiled Haskell, not bash
  # ══════════════════════════════════════════════════════════════════════════════

  fhs-run =
    if final.stdenv.isLinux then
      final.straylight.script.compiled.fhs-run
    else
      platform-stub.mk-platform-stub {
        name = "fhs-run";
        description = "Run commands in a minimal FHS namespace";
        requires = "Linux with user namespaces and bubblewrap";
        alternatives = [
          "Use Docker: docker run --rm -v $(pwd):/work -w /work <image> <cmd>"
          "Use nix-shell with FHS userenv"
          "Run directly without namespace isolation"
        ];
      };

  gpu-run =
    if final.stdenv.isLinux then
      final.straylight.script.compiled.gpu-run
    else
      platform-stub.mk-platform-stub {
        name = "gpu-run";
        description = "Run commands in a namespace with GPU device access";
        requires = "Linux with NVIDIA driver and bubblewrap";
        alternatives = [
          "Use Docker with --gpus flag: docker run --gpus all ..."
          "Use cloud instances with GPU (AWS p4, GCP A2, Lambda Labs)"
          "Run directly on a Linux machine with NVIDIA driver"
        ];
      };

in
{
  straylight = (prev.straylight or { }) // {
    container = {
      # Library functions (re-exported for convenience)
      lib = straylight-lib;

      # From namespace.nix (Linux only, with stubs)
      mk-namespace-env = namespace-mod.mk-namespace-env or (throw "mk-namespace-env requires Linux");

      # From oci.nix (Linux only, with stubs)
      mk-oci-rootfs = oci-mod.mk-oci-rootfs or (throw "mk-oci-rootfs requires Linux");
      oci-run = oci-mod.oci-run or (throw "oci-run requires Linux");

      # From firecracker.nix (Linux only)
      mk-firecracker-image =
        firecracker-mod.mk-firecracker-image or (throw "mk-firecracker-image requires Linux");

      # From extract.nix (cross-platform)
      inherit (extract-mod) extract mk-stub;

      # From ngc.nix (Linux only)
      mk-ngc-python = ngc-mod.mk-ngc-python or (throw "mk-ngc-python requires Linux");

      # From pep503.nix (cross-platform)
      inherit (pep503-mod) mk-simple-index;

      # Local runners (with platform stubs)
      inherit fhs-run gpu-run;

      # Platform info
      platform = {
        supported = final.stdenv.isLinux;
        inherit (final.stdenv.hostPlatform) system;
        reason =
          if final.stdenv.isLinux then
            "All features available"
          else
            "Container features require Linux (namespaces, bwrap, Firecracker)";
      };
    };
  };
}
