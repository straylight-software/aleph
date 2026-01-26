# nix/overlays/armitage.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                           // armitage //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   Armitage routes around the daemon. Named for the man who put the team
#   together - the one who orchestrated the run before anyone knew what
#   the real job was.
#
#   Components:
#     - armitage: Main CLI (build, store, cas operations)
#     - armitage-proxy: TLS MITM witness proxy for build fetches
#
#   The daemon is hostile infrastructure. This is the replacement.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
final: prev:
let
  inherit (prev) lib;

  # Source directory
  armitage-src = ../../src/armitage;

  # GHC 9.12 with all required packages
  hs-pkgs = final.haskell.packages.ghc912;

  # Dependencies for armitage modules
  armitage-deps =
    p: with p; [
      # Core
      text
      bytestring
      containers
      directory
      filepath
      process
      unix
      time
      aeson

      # Crypto
      crypton
      memory

      # Network
      network

      # gRPC (NativeLink CAS)
      grapesy
    ];

  # Dependencies for proxy (additional TLS)
  proxy-deps =
    p:
    armitage-deps p
    ++ (with p; [
      tls
      crypton-x509
      crypton-x509-store
      data-default-class
      pem
      asn1-types
      asn1-encoding
      hourglass
    ]);

  # GHC with armitage deps
  ghc-with-armitage = hs-pkgs.ghcWithPackages armitage-deps;

  # GHC with proxy deps
  ghc-with-proxy = hs-pkgs.ghcWithPackages proxy-deps;

  # Build armitage CLI
  mk-armitage = final.stdenv.mkDerivation {
    name = "armitage";
    src = armitage-src;

    "nativeBuildInputs" = [ ghc-with-armitage ];

    "buildPhase" = ''
      runHook preBuild
      ghc -O2 -Wall -Wno-unused-imports \
        -hidir . -odir . \
        -i. \
        -o armitage Main.hs
      runHook postBuild
    '';

    "installPhase" = ''
      runHook preInstall
      mkdir -p $out/bin
      cp armitage $out/bin/
      runHook postInstall
    '';

    meta = {
      description = "Daemon-free Nix operations with coeffect tracking";
      mainProgram = "armitage";
    };
  };

  # Build armitage-proxy
  mk-armitage-proxy = final.stdenv.mkDerivation {
    name = "armitage-proxy";
    src = armitage-src;

    "nativeBuildInputs" = [ ghc-with-proxy ];

    "buildPhase" = ''
      runHook preBuild
      ghc -O2 -Wall -Wno-unused-imports -threaded \
        -hidir . -odir . \
        -i. \
        -o armitage-proxy ProxyMain.hs
      runHook postBuild
    '';

    "installPhase" = ''
      runHook preInstall
      mkdir -p $out/bin
      cp armitage-proxy $out/bin/
      runHook postInstall
    '';

    meta = {
      description = "TLS MITM witness proxy for build fetches";
      mainProgram = "armitage-proxy";
    };
  };

in
{
  armitage = (prev.armitage or { }) // {
    # Source paths
    src = {
      root = armitage-src;
      dhall = armitage-src + "/dhall";
      proto = armitage-src + "/proto";
    };

    # GHC environments for development
    ghc = ghc-with-armitage;
    ghc-proxy = ghc-with-proxy;
  };

  # Top-level packages
  armitage-cli = mk-armitage;
  armitage-proxy = mk-armitage-proxy;
}
