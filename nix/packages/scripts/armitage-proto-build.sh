runHook pre-build

mkdir -p proto out

# Copy proto files we need
cp -r @remoteApis@/build proto/
cp -r @googleApis@/google proto/

# Create output directory structure
mkdir -p out/Proto/Build/Bazel/Remote/Execution/V2
mkdir -p out/Proto/Google/Bytestream
mkdir -p out/Proto/Google/Rpc
mkdir -p out/Proto/Google/Protobuf

# Generate Haskell bindings
protoc \
  --plugin=protoc-gen-haskell=@protoLensProtoc@/bin/proto-lens-protoc \
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
