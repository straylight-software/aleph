# nix/packages/aleph-exec.nix
#
# Zero-Bash Build Executor
# ========================
#
# This is the builder binary for zero-bash derivations. It reads a JSON spec
# and executes typed actions directly - no shell, no bash.
#
# Used as:
#   derivation {
#     builder = "${aleph-exec}/bin/aleph-exec";
#     args = [ "--spec" "${specFile}" ];
#   }
#
{
  lib,
  stdenv,
  haskellPackages,
  makeWrapper,
  patchelf,
}:

let
  # GHC with all required packages
  ghcWithPkgs = haskellPackages.ghcWithPackages (
    ps: with ps; [
      aeson
      bytestring
      containers
      directory
      filepath
      process
      text
      unix
      zip-archive
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

  # Runtime dependencies
  buildInputs = [ patchelf ];

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

    # Wrap to ensure patchelf is available at runtime
    wrapProgram $out/bin/aleph-exec \
      --prefix PATH : ${lib.makeBinPath [ patchelf ]}

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
