-- nix/modules/flake/nv-sdk/cudnn.pc.dhall
--
-- pkg-config file for cuDNN
-- Environment variables are injected by render.dhall-with-vars

let prefix : Text = env:PREFIX as Text
let cudnnVersion : Text = env:CUDNN_VERSION as Text

in ''
prefix=${prefix}
exec_prefix=${"$"}{prefix}
libdir=${"$"}{prefix}/lib
includedir=${"$"}{prefix}/include

Name: cuDNN
Description: NVIDIA CUDA Deep Neural Network library
Version: ${cudnnVersion}
Libs: -L${"$"}{libdir} -lcudnn
Cflags: -I${"$"}{includedir}
Requires: cuda
''
