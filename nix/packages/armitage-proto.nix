# nix/packages/armitage-proto.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                           // armitage-proto //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Generate Haskell bindings from Remote Execution API proto files using proto-lens.
#
# This is a build-time derivation that:
#   1. Fetches the official remote-apis protos from GitHub
#   2. Fetches required googleapis protos (status, wrappers, etc.)
#   3. Runs protoc with proto-lens-protoc plugin
#   4. Produces a Haskell library with Proto.* modules
#
# The generated modules work with grapesy for gRPC client/server.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  lib,
  prelude,
  fetch-from-git-hub,
  protobuf,
  haskell-packages,
}:
let
  # Remote Execution API protos
  remote-apis = fetch-from-git-hub {
    owner = "bazelbuild";
    repo = "remote-apis";
    rev = "v2.2.0";
    sha256 = "sha256-YlDgxZP7hLlPT2XSD7LSDZ4BuLWwWpKPAMk0E+sBwgA=";
  };

  # Google APIs (for google.rpc.Status, google.protobuf.*, etc.)
  google-apis = fetch-from-git-hub {
    owner = "googleapis";
    repo = "googleapis";
    rev = "114a745b2841a044e98cdbb19358ed29fcf4a5f1";
    sha256 = "sha256-0scS5BPVL5qhzH5LKaRKqeaMxLRQLjJAws+qUl9TJ/s=";
  };

  hs-pkgs = haskell-packages;
  inherit (hs-pkgs) ghc proto-lens-protoc;

  cabal-file = ./armitage-proto/armitage-proto.cabal;
in
prelude.stdenv.default {
  pname = "armitage-proto";
  version = "0.1.0";

  # No source - we generate everything
  dont-unpack = true;

  native-build-inputs = [
    protobuf
    proto-lens-protoc
    ghc
  ];

  build-phase =
    builtins.replaceStrings
      [ "@remoteApis@" "@googleApis@" "@protoLensProtoc@" ]
      [
        (prelude.to-string remote-apis)
        (prelude.to-string google-apis)
        (prelude.to-string proto-lens-protoc)
      ]
      (builtins.readFile ./scripts/armitage-proto-build.sh);

  install-phase = ''
    runHook preInstall

    mkdir -p $out/src
    cp -r out/* $out/src/
    cp ${cabal-file} $out/armitage-proto.cabal

    runHook postInstall
  '';

  meta = {
    description = "Proto-lens Haskell bindings for Remote Execution API";
    license = lib.licenses.asl20;
    platforms = lib.platforms.unix;
  };
}
