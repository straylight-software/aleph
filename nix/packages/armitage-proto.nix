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
  stdenv,
  fetchFromGitHub,
  protobuf,
  haskellPackages,
}:
let
  # Remote Execution API protos
  remote-apis = fetchFromGitHub {
    owner = "bazelbuild";
    repo = "remote-apis";
    rev = "v2.2.0";
    sha256 = "sha256-YlDgxZP7hLlPT2XSD7LSDZ4BuLWwWpKPAMk0E+sBwgA=";
  };

  # Google APIs (for google.rpc.Status, google.protobuf.*, etc.)
  googleapis = fetchFromGitHub {
    owner = "googleapis";
    repo = "googleapis";
    rev = "114a745b2841a044e98cdbb19358ed29fcf4a5f1";
    sha256 = "sha256-0scS5BPVL5qhzH5LKaRKqeaMxLRQLjJAws+qUl9TJ/s=";
  };

  hs-pkgs = haskellPackages;
  ghc = hs-pkgs.ghc;
  proto-lens-protoc = hs-pkgs.proto-lens-protoc;
in
stdenv.mkDerivation {
  pname = "armitage-proto";
  version = "0.1.0";

  # No source - we generate everything
  dontUnpack = true;

  nativeBuildInputs = [
    protobuf
    proto-lens-protoc
    ghc
  ];

  buildPhase = ''
    runHook preBuild

    mkdir -p proto out

    # Copy proto files we need
    cp -r ${remote-apis}/build proto/
    cp -r ${googleapis}/google proto/

    # Create output directory structure
    mkdir -p out/Proto/Build/Bazel/Remote/Execution/V2
    mkdir -p out/Proto/Google/Bytestream
    mkdir -p out/Proto/Google/Rpc
    mkdir -p out/Proto/Google/Protobuf

    # Generate Haskell bindings
    protoc \
      --plugin=protoc-gen-haskell=${proto-lens-protoc}/bin/proto-lens-protoc \
      --haskell_out=out \
      --proto_path=proto \
      proto/build/bazel/remote/execution/v2/remote_execution.proto \
      proto/google/bytestream/bytestream.proto \
      proto/google/rpc/status.proto \
      proto/google/protobuf/any.proto \
      proto/google/protobuf/duration.proto \
      proto/google/protobuf/timestamp.proto \
      proto/google/protobuf/wrappers.proto

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/src
    cp -r out/* $out/src/

    # Create a cabal file
    cat > $out/armitage-proto.cabal << 'EOF'
    cabal-version: 2.4
    name:          armitage-proto
    version:       0.1.0
    synopsis:      Proto-lens bindings for Remote Execution API
    license:       Apache-2.0
    build-type:    Simple

    library
      exposed-modules:
        Proto.Build.Bazel.Remote.Execution.V2.RemoteExecution
        Proto.Build.Bazel.Remote.Execution.V2.RemoteExecution_Fields
        Proto.Google.Bytestream.Bytestream
        Proto.Google.Bytestream.Bytestream_Fields
        Proto.Google.Rpc.Status
        Proto.Google.Rpc.Status_Fields
      hs-source-dirs: src
      default-language: Haskell2010
      build-depends:
        base >= 4.15 && < 5,
        proto-lens,
        proto-lens-runtime,
        text,
        bytestring,
        vector,
        containers,
        deepseq,
        lens-family
    EOF

    runHook postInstall
  '';

  meta = with lib; {
    description = "Proto-lens Haskell bindings for Remote Execution API";
    license = licenses.asl20;
    platforms = platforms.unix;
  };
}
