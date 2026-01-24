-- nix/modules/flake/nv-sdk/tensorrt.pc.dhall
--
-- pkg-config file for TensorRT
-- Environment variables are injected by render.dhall-with-vars

let prefix : Text = env:PREFIX as Text
let tensorrtVersion : Text = env:TENSORRT_VERSION as Text

in ''
prefix=${prefix}
exec_prefix=${"$"}{prefix}
libdir=${"$"}{prefix}/lib
includedir=${"$"}{prefix}/include

Name: TensorRT
Description: NVIDIA TensorRT inference library
Version: ${tensorrtVersion}
Libs: -L${"$"}{libdir} -lnvinfer -lnvinfer_plugin -lnvonnxparser
Cflags: -I${"$"}{includedir}
Requires: cuda cudnn
''
