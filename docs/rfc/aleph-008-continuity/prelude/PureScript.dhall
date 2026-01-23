--| PureScript Rules
--|
--| purescript_library, purescript_bundle

let Types = ./Types.dhall
let Toolchain = ./Toolchain.dhall

-- =============================================================================
-- PureScript-specific Types
-- =============================================================================

let Backend =
      < JS         -- Standard JavaScript output
      | CoreFn     -- CoreFn for alternative backends
      | ESM        -- ES Modules
      >

let Bundler =
      < Esbuild
      | Spago
      | None
      >

-- =============================================================================
-- purescript_library
-- =============================================================================

let PureScriptLibrary =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , ffi : List Text              -- FFI JavaScript files
      , backend : Backend
      , pursflags : List Text
      , toolchain : Optional Toolchain.Toolchain
      }

let purescript_library =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , ffi = [] : List Text
        , backend = Backend.JS
        , pursflags = [] : List Text
        , toolchain = None Toolchain.Toolchain
        } : PureScriptLibrary

-- =============================================================================
-- purescript_bundle (for producing runnable JS)
-- =============================================================================

let PureScriptBundle =
      { common : Types.CommonAttrs
      , entry : Text                 -- Entry module
      , deps : List Types.Dep
      , bundler : Bundler
      , minify : Bool
      , platform : < Browser | Node >
      , toolchain : Optional Toolchain.Toolchain
      }

let purescript_bundle =
      \(name : Text) ->
      \(entry : Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , entry
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , bundler = Bundler.Esbuild
        , minify = False
        , platform = < Browser | Node >.Node
        , toolchain = None Toolchain.Toolchain
        } : PureScriptBundle

-- =============================================================================
-- purescript_test
-- =============================================================================

let PureScriptTest =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , main : Text
      , toolchain : Optional Toolchain.Toolchain
      }

let purescript_test =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , main = "Test.Main"
        , toolchain = None Toolchain.Toolchain
        } : PureScriptTest

-- =============================================================================
-- Exports
-- =============================================================================

in  { Backend
    , Bundler
    , PureScriptLibrary
    , purescript_library
    , PureScriptBundle
    , purescript_bundle
    , PureScriptTest
    , purescript_test
    }
