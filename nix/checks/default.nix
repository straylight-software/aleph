# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         ALEPH-NAUGHT TEST SUITE                            ║
# ║                                                                            ║
# ║  Aggregator for all checks in aleph-naught.                                ║
# ║  Run with: nix flake check                                                 ║
# ║  Or individually: nix build .#checks.<system>.test-<name>                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝
{
  pkgs,
  system,
  lib,
  ...
}:
let
  # Import test suites
  packageTests = import ./packages.nix { inherit pkgs system lib; };
  libTests = import ./lib.nix { inherit pkgs system lib; };
  moduleTests = import ./modules.nix { inherit pkgs system lib; };
  moduleExportTests = import ./module-exports.nix { inherit pkgs system lib; };
in
# Merge all test suites into a single attrset
packageTests // libTests // moduleTests // moduleExportTests
