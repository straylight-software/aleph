-- nix/modules/flake/build/scripts/buckconfig-nv.dhall
--
-- NVIDIA toolchain configuration for Buck2
-- Environment variables are injected by render.dhall-with-vars

let nvidia_sdk_path : Text = env:NVIDIA_SDK_PATH as Text
let nvidia_sdk_include : Text = env:NVIDIA_SDK_INCLUDE as Text
let nvidia_sdk_lib : Text = env:NVIDIA_SDK_LIB as Text
let archs : Text = env:ARCHS as Text

in ''

[nv]
nvidia_sdk_path = ${nvidia_sdk_path}
nvidia_sdk_include = ${nvidia_sdk_include}
nvidia_sdk_lib = ${nvidia_sdk_lib}
archs = ${archs}
''
