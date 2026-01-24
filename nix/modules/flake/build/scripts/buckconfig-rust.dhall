-- nix/modules/flake/build/scripts/buckconfig-rust.dhall
--
-- Rust toolchain configuration for Buck2
-- Environment variables are injected by render.dhall-with-vars

let rustc : Text = env:RUSTC as Text
let rustdoc : Text = env:RUSTDOC as Text
let clippy_driver : Text = env:CLIPPY_DRIVER as Text
let cargo : Text = env:CARGO as Text

in ''

[rust]
rustc = ${rustc}
rustdoc = ${rustdoc}
clippy_driver = ${clippy_driver}
cargo = ${cargo}
target_triple = x86_64-unknown-linux-gnu
''
