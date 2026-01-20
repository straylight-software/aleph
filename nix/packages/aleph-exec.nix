# nix/packages/aleph-exec.nix
#
# Zero-Bash Build Executor
# ========================
#
# This is the builder binary for zero-bash derivations. It reads a Dhall spec
# and executes typed actions directly - no shell, no bash.
#
# DHALL IS THE SUBSTRATE.
#
# Used as:
#   derivation {
#     builder = "${aleph-exec}/bin/aleph-exec";
#     args = [ "--spec" "${specFile.dhall}" ];
#   }
#
{
  lib,
  stdenv,
  haskellPackages,
  makeWrapper,
  patchelf,
  cmake,
  ninja,
}:

let
  # GHC with all required packages including Dhall
  ghcWithPkgs = haskellPackages.ghcWithPackages (
    ps: with ps; [
      # Dhall - the substrate
      dhall

      # Core
      bytestring
      containers
      directory
      filepath
      process
      text
      unix

      # Archives
      zip-archive
      tar
      zlib

      # Globbing
      Glob
    ]
  );
in
stdenv.mkDerivation {
  pname = "aleph-exec";
  version = "0.1.0";

  src = ../scripts/Aleph/Exec;

  nativeBuildInputs = [
    ghcWithPkgs
    makeWrapper
  ];

  # Don't let stdenv think this is a CMake project
  dontUseCmakeConfigure = true;

  # These are only for runtime wrapping, not build-time
  runtimeDeps = [
    patchelf
    cmake
    ninja
  ];

  buildPhase = ''
    runHook preBuild

    # Compile
    ghc -O2 Main.hs -o aleph-exec

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin
    install -m 0755 aleph-exec $out/bin/

    # Wrap to ensure build tools are available at runtime
    wrapProgram $out/bin/aleph-exec \
      --prefix PATH : ${
        lib.makeBinPath [
          patchelf
          cmake
          ninja
        ]
      }

    runHook postInstall
  '';

  meta = with lib; {
    description = "Zero-bash build executor for typed derivations";
    homepage = "https://github.com/straylight-software/aleph";
    license = licenses.mit;
    platforms = platforms.unix;
    mainProgram = "aleph-exec";
  };
}
