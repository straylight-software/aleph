-- nix/modules/flake/build/scripts/buckconfig-lean.dhall
--
-- Lean toolchain configuration for Buck2
-- Environment variables are injected by render.dhall-with-vars

let lean : Text = env:LEAN as Text
let leanc : Text = env:LEANC as Text
let lake : Text = env:LAKE as Text
let lean_lib_dir : Text = env:LEAN_LIB_DIR as Text
let lean_include_dir : Text = env:LEAN_INCLUDE_DIR as Text

in ''

[lean]
lean = ${lean}
leanc = ${leanc}
lake = ${lake}
lean_lib_dir = ${lean_lib_dir}
lean_include_dir = ${lean_include_dir}
''
