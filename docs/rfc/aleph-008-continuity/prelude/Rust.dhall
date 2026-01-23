--| Rust Rules
--|
--| rust_library, rust_binary, rust_test

let Types = ./Types.dhall
let Toolchain = ./Toolchain.dhall

-- =============================================================================
-- Rust-specific Types
-- =============================================================================

let Edition = < Edition2018 | Edition2021 | Edition2024 >

let CrateType =
      < Bin
      | Lib
      | RLib
      | DyLib
      | CDyLib
      | StaticLib
      | ProcMacro
      >

-- =============================================================================
-- rust_library
-- =============================================================================

let RustLibrary =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , edition : Edition
      , crate_type : CrateType
      , crate_name : Optional Text
      , features : List Text
      , rustc_flags : List Types.Flag
      , proc_macro : Bool
      , toolchain : Optional Toolchain.Toolchain
      }

let rust_library =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , edition = Edition.Edition2024
        , crate_type = CrateType.RLib
        , crate_name = None Text
        , features = [] : List Text
        , rustc_flags = [] : List Types.Flag
        , proc_macro = False
        , toolchain = None Toolchain.Toolchain
        } : RustLibrary

-- =============================================================================
-- rust_binary
-- =============================================================================

let RustBinary =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , edition : Edition
      , crate_name : Optional Text
      , features : List Text
      , rustc_flags : List Types.Flag
      , link_style : < Static | Shared >
      , toolchain : Optional Toolchain.Toolchain
      }

let rust_binary =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , edition = Edition.Edition2024
        , crate_name = None Text
        , features = [] : List Text
        , rustc_flags = [] : List Types.Flag
        , link_style = < Static | Shared >.Static
        , toolchain = None Toolchain.Toolchain
        } : RustBinary

-- =============================================================================
-- rust_test
-- =============================================================================

let RustTest =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , edition : Edition
      , crate_name : Optional Text
      , features : List Text
      , rustc_flags : List Types.Flag
      , toolchain : Optional Toolchain.Toolchain
      }

let rust_test =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , edition = Edition.Edition2024
        , crate_name = None Text
        , features = [] : List Text
        , rustc_flags = [] : List Types.Flag
        , toolchain = None Toolchain.Toolchain
        } : RustTest

-- =============================================================================
-- Exports
-- =============================================================================

in  { Edition
    , CrateType
    , RustLibrary
    , rust_library
    , RustBinary
    , rust_binary
    , RustTest
    , rust_test
    }
