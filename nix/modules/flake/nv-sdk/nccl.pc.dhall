-- nix/modules/flake/nv-sdk/nccl.pc.dhall
--
-- pkg-config file for NCCL
-- Environment variables are injected by render.dhall-with-vars

let prefix : Text = env:PREFIX as Text
let ncclVersion : Text = env:NCCL_VERSION as Text

in ''
prefix=${prefix}
exec_prefix=${"$"}{prefix}
libdir=${"$"}{prefix}/lib
includedir=${"$"}{prefix}/include

Name: NCCL
Description: NVIDIA Collective Communications Library
Version: ${ncclVersion}
Libs: -L${"$"}{libdir} -lnccl
Cflags: -I${"$"}{includedir}
Requires: cuda
''
