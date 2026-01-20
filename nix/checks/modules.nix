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

      echo "Compiling all Aleph modules..."
      echo ""

      # Create temp directory for build artifacts
      mkdir -p build

      # Use --make to compile all modules with automatic dependency resolution
      # We compile the "top-level" modules that pull in everything else:
      # - Aleph.Script.Tools (imports all tool wrappers)
      # - Aleph.Script.Vm (imports Vfio, Oci, Config)
      # - Aleph.Nix (imports Types, Value, FFI)
      # - Aleph.Nix.Syntax (imports Derivation, CMake)

      ghc --make -Wall -Wno-unused-imports \
        -hidir build -odir build \
        -i$src \
        $src/Aleph/Script.hs \
        $src/Aleph/Script/Tools.hs \
        $src/Aleph/Script/Vm.hs \
        $src/Aleph/Script/Oci.hs \
        $src/Aleph/Nix.hs \
        $src/Aleph/Nix/Syntax.hs \
        2>&1 || {
          echo ""
          echo "FAILED: Module compilation failed"
          exit 1
        }

      echo ""
      echo "All Aleph modules compiled successfully"

      runHook postBuild
    '';

    installPhase = ''
      mkdir -p $out
      echo "SUCCESS" > $out/SUCCESS
      echo "All Aleph modules compiled successfully" >> $out/SUCCESS
    '';
  };

  # ==============================================================================
  # TEST: aleph-compiled-scripts
  # ==============================================================================
  # Verify all compiled scripts in straylight.script.compiled build successfully

  test-aleph-compiled-scripts = pkgs.runCommand "test-aleph-compiled-scripts" { } ''
    echo "Verifying compiled Aleph scripts..."
    echo ""

    # Check that key binaries exist and are executable
    ${lib.concatMapStringsSep "\n"
      (name: ''
        echo "  Checking ${name}..."
        if [ ! -x "${pkgs.straylight.script.compiled.${name}}/bin/${name}" ]; then
          echo "FAILED: ${name} not found or not executable"
          exit 1
        fi
        echo "    ${pkgs.straylight.script.compiled.${name}}/bin/${name}"
      '')
      [
        "vfio-bind"
        "vfio-unbind"
        "vfio-list"
        "oci-run"
        "oci-gpu"
        "oci-inspect"
        "oci-pull"
        "fhs-run"
        "gpu-run"
        "fc-run"
        "fc-build"
        "ch-run"
        "ch-gpu"
      ]
    }

    mkdir -p $out
    echo "SUCCESS" > $out/SUCCESS
    echo "All compiled Aleph scripts verified" >> $out/SUCCESS
  '';

in
# Only run on Linux (Aleph.Nix has FFI bindings that may need Linux)
lib.optionalAttrs (system == "x86_64-linux" || system == "aarch64-linux") {
  inherit
    test-aleph-modules
    test-aleph-compiled-scripts
    ;
}
