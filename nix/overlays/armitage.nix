# nix/overlays/armitage.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                           // armitage //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   Armitage is the shim between typed builds and the nix daemon. Named for
#   the man who put the team together - the one who orchestrated the run
#   before anyone knew what the real job was.
#
#   - armitage-proxy: Witness proxy for build-time network access
#   - (future) coeffect checker, attestation tools
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
final: prev:
let
  inherit (prev) lib;

  # Source directory for armitage
  armitage-src = ../../armitage;
  proxy-src = armitage-src + "/proxy/src";

  # Use GHC 9.12 consistently
  hs-pkgs = final.haskell.packages.ghc912;

  # Dependencies for armitage-proxy
  # Network server + cryptography + JSON
  proxy-deps =
    p: with p; [
      text
      aeson
      bytestring
      directory
      filepath
      network
      crypton
      memory
      time
      containers
    ];

  # GHC with proxy dependencies
  ghc-with-proxy = hs-pkgs.ghcWithPackages proxy-deps;

in
{
  armitage = (prev.armitage or { }) // {
    # ──────────────────────────────────────────────────────────────────────────
    # // witness proxy //
    # ──────────────────────────────────────────────────────────────────────────
    #
    # HTTP/HTTPS proxy that witnesses all build-time network fetches.
    #
    #   - Caches responses by SHA256 (content-addressed)
    #   - Logs all fetches as JSONL attestations
    #   - Enforces domain allowlist
    #   - Tunnels HTTPS via CONNECT method
    #
    # Usage:
    #   nix build .#armitage-proxy
    #   PROXY_PORT=8080 ./result/bin/armitage-proxy
    #
    # Configure builds:
    #   HTTP_PROXY=http://proxy:8080 HTTPS_PROXY=http://proxy:8080 nix build ...

    proxy = final.stdenv.mkDerivation {
      name = "armitage-proxy";
      src = proxy-src;
      "dontUnpack" = true;

      "nativeBuildInputs" = [ ghc-with-proxy ];

      "buildPhase" = ''
        runHook preBuild
        ghc -O2 -Wall -Wno-unused-imports \
          -threaded -rtsopts "-with-rtsopts=-N" \
          -hidir . -odir . \
          -o armitage-proxy ${proxy-src}/Main.hs
        runHook postBuild
      '';

      "installPhase" = ''
        runHook preInstall
        mkdir -p $out/bin
        cp armitage-proxy $out/bin/
        runHook postInstall
      '';

      meta = {
        description = "Armitage Witness Proxy - Content-addressed caching HTTP proxy";
        mainProgram = "armitage-proxy";
      };
    };

    # GHC with armitage dependencies (for development)
    ghc = ghc-with-proxy;
  };
}
