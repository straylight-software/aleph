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
final: prev: {
  # Elan - Lean version manager (like rustup for Rust)
  # This allows lake to download the correct Lean version for each project
  elan =
    prev.elan or (prev.callPackage (
      {
        lib,
        stdenv,
        fetchFromGitHub,
        rustPlatform,
      }:
      rustPlatform.buildRustPackage rec {
        pname = "elan";
        version = "4.1.2";

        src = fetchFromGitHub {
          owner = "leanprover";
          repo = "elan";
          rev = "v${version}";
          hash = "sha256-abc123"; # Would need real hash
        };

        cargoHash = "sha256-xyz789"; # Would need real hash

        meta = with lib; {
          description = "Lean version manager";
          homepage = "https://github.com/leanprover/elan";
          license = licenses.asl20;
          platforms = platforms.unix;
        };
      }
    ) { });

  # lean4-mathlib-env - Lean 4 environment with mathlib cache
  # Uses elan to fetch the correct toolchain
  lean4-mathlib-env = prev.buildEnv {
    name = "lean4-mathlib-env";
    paths = [
      final.elan
      prev.git
      prev.curl
      prev.cacert
    ];
    pathsToLink = [ "/bin" ];
  };
}
