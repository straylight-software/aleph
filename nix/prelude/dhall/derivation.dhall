-- nix/prelude/dhall/derivation.dhall
--
-- Complete derivation schema for typed packages.
--
-- This is the contract between Haskell WASM plugins and Nix. The WASM plugin
-- emits a Dhall expression conforming to this schema. Nix validates all
-- StorePaths against the actual store, then creates a derivation with
-- aleph-exec as the builder.

let StorePath = ./store-path.dhall
let Actions = ./actions.dhall

-- Source specification
let Src =
  < GitHub : { owner : Text, repo : Text, rev : Text, hash : Text }
  | Url : { url : Text, hash : Text }
  | Store : StorePath.StorePath
  | None
  >

-- Build system configuration
let Builder =
  < CMake : { flags : List Text, buildType : Optional Text }
  | Autotools : { configureFlags : List Text, makeFlags : List Text }
  | Meson : { flags : List Text }
  | Custom : { buildPhase : Actions.Phase, installPhase : Actions.Phase }
  | None
  >

-- Package metadata
let Meta : Type =
  { description : Text
  , homepage : Optional Text
  , license : Text
  , platforms : List Text
  , mainProgram : Optional Text
  }

let defaultMeta : Meta =
  { description = ""
  , homepage = None Text
  , license = "unfree"
  , platforms = [] : List Text
  , mainProgram = None Text
  }

-- Dependencies by category
let Deps : Type =
  { nativeBuildInputs : List Text    -- Build-time tools (resolved to StorePath by Nix)
  , buildInputs : List Text          -- Libraries to link
  , propagatedBuildInputs : List Text
  , checkInputs : List Text
  }

let emptyDeps : Deps =
  { nativeBuildInputs = [] : List Text
  , buildInputs = [] : List Text
  , propagatedBuildInputs = [] : List Text
  , checkInputs = [] : List Text
  }

-- Build phases (typed actions)
let Phases : Type =
  { postPatch : Actions.Phase
  , preConfigure : Actions.Phase
  , installPhase : Actions.Phase
  , postInstall : Actions.Phase
  , postFixup : Actions.Phase
  }

let emptyPhases : Phases =
  { postPatch = [] : Actions.Phase
  , preConfigure = [] : Actions.Phase
  , installPhase = [] : Actions.Phase
  , postInstall = [] : Actions.Phase
  , postFixup = [] : Actions.Phase
  }

-- Environment variables
let EnvVar : Type = { name : Text, value : Text }

-- The complete derivation specification
let Derivation : Type =
  { pname : Text
  , version : Text
  , src : Src
  , builder : Builder
  , deps : Deps
  , phases : Phases
  , meta : Meta
  , env : List EnvVar
  , patches : List StorePath.StorePath
  , strictDeps : Bool
  , doCheck : Bool
  , dontUnpack : Bool
  }

let defaultDerivation : Derivation =
  { pname = ""
  , version = ""
  , src = Src.None
  , builder = Builder.None
  , deps = emptyDeps
  , phases = emptyPhases
  , meta = defaultMeta
  , env = [] : List EnvVar
  , patches = [] : List StorePath.StorePath
  , strictDeps = True
  , doCheck = False
  , dontUnpack = False
  }

-- Convenience: create a derivation with just the required fields
let mkDerivation
  : Text -> Text -> Src -> Derivation
  = \(pname : Text) ->
    \(version : Text) ->
    \(src : Src) ->
      defaultDerivation // { pname, version, src }

in
{ StorePath = StorePath.StorePath
, Action = Actions.Action
, Phase = Actions.Phase
, WrapAction = Actions.WrapAction
, Replacement = Actions.Replacement
, Src
, Builder
, Meta
, Deps
, Phases
, EnvVar
, Derivation
-- Defaults
, defaultMeta
, emptyDeps
, emptyPhases
, defaultDerivation
, mkDerivation
-- Re-export action constructors
, mkdir = Actions.mkdir
, copy = Actions.copy
, symlink = Actions.symlink
, writeFile = Actions.writeFile
, install = Actions.install
, installBin = Actions.installBin
, installLib = Actions.installLib
, installHeader = Actions.installHeader
, installData = Actions.installData
, remove = Actions.remove
, unzip = Actions.unzip
, patchElfRpath = Actions.patchElfRpath
, substitute = Actions.substitute
, replace = Actions.replace
, wrap = Actions.wrap
, wrapPrefix = Actions.wrapPrefix
, wrapSet = Actions.wrapSet
, chmod = Actions.chmod
-- Re-export store path helpers
, pkg = StorePath.pkg
, withOutput = StorePath.withOutput
, lib = StorePath.lib
, bin = StorePath.bin
, include = StorePath.include
, share = StorePath.share
}
