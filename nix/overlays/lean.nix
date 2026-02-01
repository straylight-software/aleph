# overlays/lean.nix
#
# Lean 4 overlay with mathlib and ecosystem support.
#
# Provides:
#   - pkgs.lean4 - Latest Lean 4 from leanprover releases
#   - pkgs.elan - Lean version manager
#   - pkgs.lean4WithPackages - Lean 4 with prebuilt mathlib cache
#
# For projects needing mathlib, we use lake with elan to handle
# toolchain version requirements automatically.
#
# NOTE: This overlay must be applied after the prelude overlay.
#
final: prev:
let
  inherit (prev.aleph) build-env;

  # Lisp-case wrappers for nixpkgs functions
  # Uses get' (from prelude) to avoid camelCase identifiers in code
  # TODO: need a aleph.build-rust-package wrapper
in
{
  # Elan - Lean version manager (like rustup for Rust)
  # This allows lake to download the correct Lean version for each project
  # NOTE: Using prev.elan from nixpkgs (no custom build needed)
  inherit (prev) elan;

  # lean4-mathlib-env - Lean 4 environment with mathlib cache
  # Uses elan to fetch the correct toolchain
  lean4-mathlib-env = build-env {
    name = "lean4-mathlib-env";
    paths = [
      final.elan
      prev.git
      prev.curl
      prev.cacert
    ];
    paths-to-link = [ "/bin" ];
  };
}
