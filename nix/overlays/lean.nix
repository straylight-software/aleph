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
  inherit (prev) lib;
  inherit (prev.straylight) prelude translate-attrs;
  inherit (prelude) licenses platforms get';

  # Lisp-case wrappers for nixpkgs functions with attribute translation
  # Uses get' (from prelude) to avoid camelCase identifiers in code
  call-package = get' "callPackage" prev;
  fetch-from-github = get' "fetchFromGitHub" prev;
  rust-platform = get' "rustPlatform" prev;
  build-rust-package = f: (get' "buildRustPackage" rust-platform) (attrs: translate-attrs (f attrs));
  build-env = attrs: (get' "buildEnv" prev) (translate-attrs attrs);
in
{
  # Elan - Lean version manager (like rustup for Rust)
  # This allows lake to download the correct Lean version for each project
  elan =
    prev.elan or (call-package (
      { stdenv }:
      build-rust-package (attrs: {
        pname = "elan";
        version = "4.1.2";

        src = fetch-from-github {
          owner = "leanprover";
          repo = "elan";
          rev = "v${attrs.version}";
          hash = "sha256-abc123"; # Would need real hash
        };

        cargo-hash = "sha256-xyz789"; # Would need real hash

        meta = {
          description = "Lean version manager";
          homepage = "https://github.com/leanprover/elan";
          license = licenses.asl20;
          platforms = platforms.unix;
        };
      })
    ) { });

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
