# nix/packages/aleph-build.nix
#
# aleph-build: Unified build dispatcher
#
# This is the builder binary that reads BuildContext.dhall and dispatches
# to the appropriate build logic. No env vars except ALEPH_CONTEXT.
#
{
  lib,
  stdenv,
  haskellPackages,
  makeWrapper,
  cmake,
  ninja,
  gnumake,
  meson,
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
    ]
  );

  # Path to Aleph.Build library
  alephScripts = ../scripts;
  alephBuilders = ../builders;
in
stdenv.mkDerivation {
  pname = "aleph-build";
  version = "0.1.0";

  # Source is the builder and library
  src = alephBuilders;

  nativeBuildInputs = [
    ghcWithPkgs
    makeWrapper
  ];

  # Don't let stdenv think this is a CMake project
  dontUseCmakeConfigure = true;

  buildPhase = ''
    runHook preBuild

    # Copy Aleph.Build library to a writable location (GHC needs to write .hi files)
    mkdir -p lib/Aleph/Build
    cp ${alephScripts}/Aleph/Build.hs lib/Aleph/
    # Copy all .hs files from Build/
    for f in ${alephScripts}/Aleph/Build/*.hs; do
      cp "$f" lib/Aleph/Build/
    done

    # Compile aleph-build with access to Aleph.Build library
    ghc -O2 \
      -ilib \
      aleph-build.hs \
      -o aleph-build

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin
    install -m 0755 aleph-build $out/bin/

    # Wrap to ensure build tools are available at runtime
    wrapProgram $out/bin/aleph-build \
      --prefix PATH : ${
        lib.makeBinPath [
          cmake
          ninja
          gnumake
          meson
        ]
      }

    runHook postInstall
  '';

  meta = with lib; {
    description = "Unified build dispatcher for Aleph-1";
    homepage = "https://github.com/straylight-software/aleph";
    license = licenses.mit;
    platforms = platforms.unix;
    mainProgram = "aleph-build";
  };
}
