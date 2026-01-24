# ==============================================================================
#                         ALEPH MODULE COMPILATION TESTS
#
#  Verifies that all Aleph.* modules compile successfully.
#  This catches import errors, type errors, and missing dependencies.
#
#  We use GHC's --make mode which handles dependency ordering automatically.
# ==============================================================================
{
  pkgs,
  system,
  lib,
  ...
}:
let
  # Get script source and GHC from the overlay
  inherit (pkgs.straylight.script) src ghc;

  # ==============================================================================
  # TEST: aleph-modules
  # ==============================================================================
  # Compile all Aleph.* modules using GHC's --make mode.
  # This automatically handles dependency ordering and verifies everything compiles.

  test-aleph-modules = pkgs.stdenv.mkDerivation {
    name = "test-aleph-modules";
    inherit src;
    dontUnpack = true;

    nativeBuildInputs = [ ghc ];

    buildPhase = ''
      runHook preBuild
      ${builtins.readFile ./scripts/test-aleph-modules.bash}
      runHook postBuild
    '';

    installPhase = ''
      mkdir -p $out
      echo "SUCCESS" > $out/SUCCESS
      echo "All Aleph modules compiled successfully" >> $out/SUCCESS
    '';

    meta = {
      description = "Test that all Aleph.* Haskell modules compile successfully";
    };
  };

  # ==============================================================================
  # TEST: aleph-compiled-scripts
  # ==============================================================================
  # Verify all compiled scripts in straylight.script.compiled build successfully

  scriptNames = [
    "vfio-bind"
    "vfio-unbind"
    "vfio-list"
    "crane-inspect"
    "crane-pull"
    "unshare-run"
    "unshare-gpu"
    "fhs-run"
    "gpu-run"
    "isospin-run"
    "isospin-build"
    "cloud-hypervisor-run"
    "cloud-hypervisor-gpu"
  ];

  scriptChecks = lib.concatMapStringsSep "\n" (name: ''
    echo "  Checking ${name}..."
    if [ ! -x "${pkgs.straylight.script.compiled.${name}}/bin/${name}" ]; then
      echo "FAILED: ${name} not found or not executable"
      exit 1
    fi
    echo "    ${pkgs.straylight.script.compiled.${name}}/bin/${name}"
  '') scriptNames;

  test-aleph-compiled-scripts = pkgs.runCommand "test-aleph-compiled-scripts" { } (
    pkgs.replaceVars ./scripts/test-aleph-compiled-scripts.bash {
      inherit scriptChecks;
    }
  );

in
# Only run on Linux (Aleph.Nix has FFI bindings that may need Linux)
lib.optionalAttrs (system == "x86_64-linux" || system == "aarch64-linux") {
  inherit
    test-aleph-modules
    test-aleph-compiled-scripts
    ;
}
