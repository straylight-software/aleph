# nix/checks/module-exports.nix
#
# Tests that exported flake modules can be imported by downstream flakes.
# Verifies the public API surface works correctly.
#
{
  pkgs,
  lib,
  system,
  ...
}:
let
  # Get shortlist packages from config (they're exported via packages output)
  # These will be available as shortlist-fmt, shortlist-spdlog, etc.
  # We verify them by importing directly from the shortlist module

  # Test that the module composition works by verifying expected attributes exist
  test-shortlist-builds = pkgs.runCommand "test-shortlist-builds" { } (
    builtins.readFile ./scripts/test-shortlist-builds.bash
  );

  # Test module composition - verify 'full' module has expected submodules
  test-module-full = pkgs.runCommand "test-module-full" { } (
    builtins.readFile ./scripts/test-module-full.bash
  );

in
lib.optionalAttrs (system == "x86_64-linux" || system == "aarch64-linux") {
  inherit
    test-shortlist-builds
    test-module-full
    ;
}
