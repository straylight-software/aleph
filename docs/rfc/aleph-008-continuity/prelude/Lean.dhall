--| Lean4 Rules
--|
--| lean_library, lean_binary, lean_test
--| Plus C extraction for verified code

let Types = ./Types.dhall
let Toolchain = ./Toolchain.dhall

-- =============================================================================
-- Lean-specific Types
-- =============================================================================

let Backend =
      < Default     -- Native Lean
      | C           -- Extract to C
      | LLVM        -- Direct LLVM (experimental)
      >

-- =============================================================================
-- lean_library
-- =============================================================================

let LeanLibrary =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , root : Text                      -- Root module name
      , backend : Backend
      , extraArgs : List Text
      , toolchain : Optional Toolchain.Toolchain
      -- For C extraction
      , extract_c : Bool
      , c_deps : List Types.Dep          -- C deps for extracted code
      }

let lean_library =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , root = name
        , backend = Backend.Default
        , extraArgs = [] : List Text
        , toolchain = None Toolchain.Toolchain
        , extract_c = False
        , c_deps = [] : List Types.Dep
        } : LeanLibrary

-- =============================================================================
-- lean_binary
-- =============================================================================

let LeanBinary =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , main : Text                      -- Main module
      , backend : Backend
      , extraArgs : List Text
      , toolchain : Optional Toolchain.Toolchain
      }

let lean_binary =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
      \(main : Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , main
        , backend = Backend.Default
        , extraArgs = [] : List Text
        , toolchain = None Toolchain.Toolchain
        } : LeanBinary

-- =============================================================================
-- lean_test (for #check, example, etc.)
-- =============================================================================

let LeanTest =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , toolchain : Optional Toolchain.Toolchain
      }

let lean_test =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , toolchain = None Toolchain.Toolchain
        } : LeanTest

-- =============================================================================
-- Proven library: Lean -> C -> native/wasm
-- =============================================================================

let ProvenLibrary =
      { common : Types.CommonAttrs
      , lean_srcs : List Text
      , lean_deps : List Types.Dep
      , root : Text
      , target : Types.Triple
      , verified : Bool                  -- Must typecheck with no sorry
      , c_flags : List Types.Flag
      }

let proven_library =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
      \(target : Types.Triple) ->
        { common = Types.defaultCommon name
        , lean_srcs = srcs
        , lean_deps = List/map Text Types.Dep Types.Dep.Target deps
        , root = name
        , target
        , verified = True
        , c_flags = [] : List Types.Flag
        } : ProvenLibrary

-- =============================================================================
-- Exports
-- =============================================================================

in  { Backend
    , LeanLibrary
    , lean_library
    , LeanBinary
    , lean_binary
    , LeanTest
    , lean_test
    , ProvenLibrary
    , proven_library
    }
