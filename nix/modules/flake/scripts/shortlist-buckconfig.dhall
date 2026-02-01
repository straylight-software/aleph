-- nix/modules/flake/scripts/shortlist-buckconfig.dhall
--
-- Hermetic C++ library paths for Buck2
-- Environment variables are injected by render.dhall-with-vars

let zlibNgLine : Text = env:ZLIB_NG_LINE as Text
let fmtLine : Text = env:FMT_LINE as Text
let fmtDevLine : Text = env:FMT_DEV_LINE as Text
let catch2Line : Text = env:CATCH2_LINE as Text
let catch2DevLine : Text = env:CATCH2_DEV_LINE as Text
let spdlogLine : Text = env:SPDLOG_LINE as Text
let spdlogDevLine : Text = env:SPDLOG_DEV_LINE as Text
let mdspanLine : Text = env:MDSPAN_LINE as Text
let rapidjsonLine : Text = env:RAPIDJSON_LINE as Text
let nlohmannJsonLine : Text = env:NLOHMANN_JSON_LINE as Text
let libsodiumLine : Text = env:LIBSODIUM_LINE as Text
let libsodiumDevLine : Text = env:LIBSODIUM_DEV_LINE as Text
let simdjsonLine : Text = env:SIMDJSON_LINE as Text

in ''

[shortlist]
# Hermetic C++ libraries
# Format: lib = /nix/store/..., lib_dev = /nix/store/...-dev (for headers)
${zlibNgLine}
${fmtLine}
${fmtDevLine}
${catch2Line}
${catch2DevLine}
${spdlogLine}
${spdlogDevLine}
${mdspanLine}
${rapidjsonLine}
${nlohmannJsonLine}
${libsodiumLine}
${libsodiumDevLine}
${simdjsonLine}
''
