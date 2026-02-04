# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         ALEPH TEST SUITE                            ║
# ║                                                                            ║
# ║  Aggregator for all checks in aleph.                                ║
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
  # :: t8
  # :: t10
  # :: t12
  # :: t14
  package-tests = import ./packages.nix { inherit pkgs system lib; };
  lib-tests = import ./lib.nix { inherit pkgs system lib; };
  module-tests = import ./modules.nix { inherit pkgs system lib; };
  module-export-tests = import ./module-exports.nix { inherit pkgs system lib; };
in
# Merge all test suites into a single attrset
package-tests // lib-tests // module-tests // module-export-tests
