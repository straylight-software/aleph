--| Build Rules
--|
--| rust_library, rust_binary, lean_library, c_library, etc.
--| No globs. Explicit file lists.

let Toolchain = ./Toolchain.dhall
let Target = ./Target.dhall
let Action = ./Action.dhall

let Artifact = Toolchain.Artifact

-- Visibility
let Visibility =
      < Public
      | Private
      | Package
      | Targets : List Text
      >

-- Common rule fields
let CommonFields =
      { name : Text
      , visibility : Visibility
      , labels : List Text
      }

let defaultCommon
    : Text -> CommonFields
    = \(name : Text) ->
        { name
        , visibility = Visibility.Public
        , labels = [] : List Text
        }

--| Rust Edition
let RustEdition =
      < Edition2018
      | Edition2021
      | Edition2024
      >

--| Rust crate type
let CrateType =
      < Bin
      | Lib
      | RLib
      | DyLib
      | CDyLib
      | StaticLib
      | ProcMacro
      >

--| Rust library rule
let RustLibrary =
      { common : CommonFields
      , srcs : List Text           -- NO GLOBS
      , deps : List Text
      , edition : RustEdition
      , crate_type : CrateType
      , features : List Text
      , rustflags : List Toolchain.Flag
      , proc_macro : Bool
      }

let rust_library
    : Text -> List Text -> List Text -> RustLibrary
    = \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = defaultCommon name
        , srcs
        , deps
        , edition = RustEdition.Edition2024
        , crate_type = CrateType.RLib
        , features = [] : List Text
        , rustflags = [] : List Toolchain.Flag
        , proc_macro = False
        }

--| Rust binary rule
let RustBinary =
      { common : CommonFields
      , srcs : List Text
      , deps : List Text
      , edition : RustEdition
      , features : List Text
      , rustflags : List Toolchain.Flag
      }

let rust_binary
    : Text -> List Text -> List Text -> RustBinary
    = \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = defaultCommon name
        , srcs
        , deps
        , edition = RustEdition.Edition2024
        , features = [] : List Text
        , rustflags = [] : List Toolchain.Flag
        }

--| Lean library rule
let LeanLibrary =
      { common : CommonFields
      , srcs : List Text
      , deps : List Text
      , root : Text                -- Root module
      , extraArgs : List Text
      }

let lean_library
    : Text -> List Text -> List Text -> LeanLibrary
    = \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = defaultCommon name
        , srcs
        , deps
        , root = name
        , extraArgs = [] : List Text
        }

--| C++ standard
let CxxStandard =
      < Cxx11
      | Cxx14
      | Cxx17
      | Cxx20
      | Cxx23
      >

--| C library rule
let CLibrary =
      { common : CommonFields
      , srcs : List Text
      , hdrs : List Text
      , deps : List Text
      , includes : List Text
      , defines : List { name : Text, value : Optional Text }
      , copts : List Toolchain.Flag
      , standard : CxxStandard
      }

let c_library
    : Text -> List Text -> List Text -> List Text -> CLibrary
    = \(name : Text) ->
      \(srcs : List Text) ->
      \(hdrs : List Text) ->
      \(deps : List Text) ->
        { common = defaultCommon name
        , srcs
        , hdrs
        , deps
        , includes = [] : List Text
        , defines = [] : List { name : Text, value : Optional Text }
        , copts = [] : List Toolchain.Flag
        , standard = CxxStandard.Cxx17
        }

--| WASM module rule
let WasmOptLevel =
      < O0 | O1 | O2 | O3 | Oz | Os >

let WasmFeature =
      < bulk_memory
      | simd128
      | threads
      | exception_handling
      | tail_call
      | multi_memory
      >

let WasmModule =
      { common : CommonFields
      , src : Text                 -- Input (compiled C/Rust)
      , optimize : WasmOptLevel
      , features : List WasmFeature
      , exports : List Text
      }

let wasm_module
    : Text -> Text -> WasmModule
    = \(name : Text) ->
      \(src : Text) ->
        { common = defaultCommon name
        , src
        , optimize = WasmOptLevel.O3
        , features = [ WasmFeature.bulk_memory, WasmFeature.simd128 ]
        , exports = [] : List Text
        }

--| Generic rule union
let Rule =
      < RustLibrary : RustLibrary
      | RustBinary : RustBinary
      | LeanLibrary : LeanLibrary
      | CLibrary : CLibrary
      | WasmModule : WasmModule
      >

in  { Visibility
    , CommonFields
    , defaultCommon
    , RustEdition
    , CrateType
    , RustLibrary
    , rust_library
    , RustBinary
    , rust_binary
    , LeanLibrary
    , lean_library
    , CxxStandard
    , CLibrary
    , c_library
    , WasmOptLevel
    , WasmFeature
    , WasmModule
    , wasm_module
    , Rule
    }
