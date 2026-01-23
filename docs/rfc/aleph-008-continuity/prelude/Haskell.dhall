--| Haskell Rules
--|
--| haskell_library, haskell_binary, haskell_test

let Types = ./Types.dhall
let Toolchain = ./Toolchain.dhall

-- =============================================================================
-- Haskell-specific Types
-- =============================================================================

let Extension =
      < OverloadedStrings
      | LambdaCase
      | TypeApplications
      | GADTs
      | DataKinds
      | TypeFamilies
      | RankNTypes
      | ScopedTypeVariables
      | FlexibleContexts
      | FlexibleInstances
      | MultiParamTypeClasses
      | DeriveGeneric
      | DeriveFunctor
      | DeriveFoldable
      | DeriveTraversable
      | GeneralizedNewtypeDeriving
      | StandaloneDeriving
      | TemplateHaskell
      | QuasiQuotes
      | BangPatterns
      | ViewPatterns
      | PatternSynonyms
      | BlockArguments
      | RecordWildCards
      | NamedFieldPuns
      | TupleSections
      | NumericUnderscores
      | ImportQualifiedPost
      | NoImplicitPrelude
      | Custom : Text
      >

-- =============================================================================
-- haskell_library
-- =============================================================================

let HaskellLibrary =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , exposed_modules : List Text
      , other_modules : List Text
      , extensions : List Extension
      , ghc_options : List Text
      , toolchain : Optional Toolchain.Toolchain
      }

let haskell_library =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , exposed_modules = [] : List Text
        , other_modules = [] : List Text
        , extensions = [] : List Extension
        , ghc_options = [] : List Text
        , toolchain = None Toolchain.Toolchain
        } : HaskellLibrary

-- =============================================================================
-- haskell_binary
-- =============================================================================

let HaskellBinary =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , main : Text
      , extensions : List Extension
      , ghc_options : List Text
      , link_style : < Static | Shared >
      , toolchain : Optional Toolchain.Toolchain
      }

let haskell_binary =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
      \(main : Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , main
        , extensions = [] : List Extension
        , ghc_options = [] : List Text
        , link_style = < Static | Shared >.Static
        , toolchain = None Toolchain.Toolchain
        } : HaskellBinary

-- =============================================================================
-- haskell_test (HSpec, Tasty, etc)
-- =============================================================================

let HaskellTest =
      { common : Types.CommonAttrs
      , srcs : List Text
      , deps : List Types.Dep
      , main : Text
      , extensions : List Extension
      , ghc_options : List Text
      , toolchain : Optional Toolchain.Toolchain
      }

let haskell_test =
      \(name : Text) ->
      \(srcs : List Text) ->
      \(deps : List Text) ->
        { common = Types.defaultCommon name
        , srcs
        , deps = List/map Text Types.Dep Types.Dep.Target deps
        , main = "Main"
        , extensions = [] : List Extension
        , ghc_options = [] : List Text
        , toolchain = None Toolchain.Toolchain
        } : HaskellTest

-- =============================================================================
-- Exports
-- =============================================================================

in  { Extension
    , HaskellLibrary
    , haskell_library
    , HaskellBinary
    , haskell_binary
    , HaskellTest
    , haskell_test
    }
