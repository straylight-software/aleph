--| Nix Derivation Types
--|
--| This is the bridge: typed Dhall that compiles to Nix derivations.
--| During transition, we can write derivations in Dhall and emit Nix.
--| Over time, we tighten: derivations become DICE actions directly.

let Types = ./Types.dhall

-- =============================================================================
-- Core Derivation Types
-- =============================================================================

let System =
      < x86_64-linux
      | aarch64-linux
      | x86_64-darwin
      | aarch64-darwin
      >

let OutputHashMode =
      < Flat        -- Hash of file contents
      | Recursive   -- Hash of NAR serialization
      >

let OutputHashAlgo =
      < SHA256
      | SHA512
      >

-- | A derivation output
let Output =
      { name : Text
      , path : Optional Text           -- Fixed output path (for FOD)
      , hashMode : Optional OutputHashMode
      , hashAlgo : Optional OutputHashAlgo
      , hash : Optional Text           -- Expected hash (for FOD)
      }

let defaultOutput : Output =
      { name = "out"
      , path = None Text
      , hashMode = None OutputHashMode
      , hashAlgo = None OutputHashAlgo
      , hash = None Text
      }

-- | A derivation input (another derivation or source)
let Input =
      < Drv : { drv : Text, outputs : List Text }  -- Another derivation
      | Src : Text                                  -- Source path
      | Store : Types.Artifact                      -- Content-addressed store path
      >

-- | Environment variable
let EnvVar =
      { name : Text
      , value : Text
      }

-- =============================================================================
-- The Derivation Record
-- =============================================================================

let Derivation =
      { name : Text
      , system : System
      , builder : Text                 -- Path to builder executable
      , args : List Text               -- Arguments to builder
      , env : List EnvVar              -- Environment variables
      , inputs : List Input            -- Input derivations/sources
      , outputs : List Output          -- Output specifications
      -- Nix-specific
      , allowedReferences : Optional (List Text)
      , allowedRequisites : Optional (List Text)
      , disallowedReferences : Optional (List Text)
      , disallowedRequisites : Optional (List Text)
      , preferLocalBuild : Bool
      , allowSubstitutes : Bool
      }

let emptyDerivation : Derivation =
      { name = ""
      , system = System.x86_64-linux
      , builder = "/bin/sh"
      , args = [] : List Text
      , env = [] : List EnvVar
      , inputs = [] : List Input
      , outputs = [ defaultOutput ]
      , allowedReferences = None (List Text)
      , allowedRequisites = None (List Text)
      , disallowedReferences = None (List Text)
      , disallowedRequisites = None (List Text)
      , preferLocalBuild = False
      , allowSubstitutes = True
      }

-- =============================================================================
-- Fixed-Output Derivations (FOD)
-- =============================================================================

let FetchUrl =
      { url : Text
      , sha256 : Text
      , name : Optional Text
      , executable : Bool
      , unpack : Bool
      }

let fetchurl =
      \(url : Text) ->
      \(sha256 : Text) ->
        { url
        , sha256
        , name = None Text
        , executable = False
        , unpack = False
        } : FetchUrl

let FetchGit =
      { url : Text
      , rev : Text
      , sha256 : Text
      , fetchSubmodules : Bool
      , deepClone : Bool
      , leaveDotGit : Bool
      }

let fetchgit =
      \(url : Text) ->
      \(rev : Text) ->
      \(sha256 : Text) ->
        { url
        , rev
        , sha256
        , fetchSubmodules = False
        , deepClone = False
        , leaveDotGit = False
        } : FetchGit

-- =============================================================================
-- stdenv-like Builders
-- =============================================================================

let BuildPhase =
      < Unpack
      | Patch
      | Configure
      | Build
      | Check
      | Install
      | Fixup
      | InstallCheck
      | Custom : Text
      >

let MkDerivation =
      { name : Text
      , version : Optional Text
      , src : Input
      , system : System
      
      -- Dependencies
      , buildInputs : List Input
      , nativeBuildInputs : List Input
      , propagatedBuildInputs : List Input
      , propagatedNativeBuildInputs : List Input
      
      -- Phases
      , phases : List BuildPhase
      , unpackPhase : Optional Text
      , patchPhase : Optional Text
      , configurePhase : Optional Text
      , buildPhase : Optional Text
      , checkPhase : Optional Text
      , installPhase : Optional Text
      , fixupPhase : Optional Text
      
      -- Flags
      , doCheck : Bool
      , dontFixup : Bool
      , dontStrip : Bool
      
      -- Outputs
      , outputs : List Text
      , meta : { description : Text, license : Text, platforms : List System }
      }

let mkDerivation =
      \(name : Text) ->
      \(src : Input) ->
        { name
        , version = None Text
        , src
        , system = System.x86_64-linux
        , buildInputs = [] : List Input
        , nativeBuildInputs = [] : List Input
        , propagatedBuildInputs = [] : List Input
        , propagatedNativeBuildInputs = [] : List Input
        , phases = [ BuildPhase.Unpack
                   , BuildPhase.Patch
                   , BuildPhase.Configure
                   , BuildPhase.Build
                   , BuildPhase.Check
                   , BuildPhase.Install
                   , BuildPhase.Fixup
                   ]
        , unpackPhase = None Text
        , patchPhase = None Text
        , configurePhase = None Text
        , buildPhase = None Text
        , checkPhase = None Text
        , installPhase = None Text
        , fixupPhase = None Text
        , doCheck = True
        , dontFixup = False
        , dontStrip = False
        , outputs = [ "out" ]
        , meta = { description = "", license = "", platforms = [ System.x86_64-linux ] }
        } : MkDerivation

-- =============================================================================
-- Rust Derivations (cargo-based)
-- =============================================================================

let RustDerivation =
      { name : Text
      , version : Text
      , src : Input
      , cargoSha256 : Text             -- Hash of Cargo.lock dependencies
      , buildType : < Release | Debug >
      , features : List Text
      , buildInputs : List Input
      , nativeBuildInputs : List Input
      , doCheck : Bool
      , meta : { description : Text, license : Text }
      }

let buildRustPackage =
      \(name : Text) ->
      \(version : Text) ->
      \(src : Input) ->
      \(cargoSha256 : Text) ->
        { name
        , version
        , src
        , cargoSha256
        , buildType = < Release | Debug >.Release
        , features = [] : List Text
        , buildInputs = [] : List Input
        , nativeBuildInputs = [] : List Input
        , doCheck = True
        , meta = { description = "", license = "" }
        } : RustDerivation

-- =============================================================================
-- Haskell Derivations (cabal-based)
-- =============================================================================

let HaskellDerivation =
      { name : Text
      , version : Text
      , src : Input
      , isLibrary : Bool
      , isExecutable : Bool
      , buildDepends : List Input
      , executableDepends : List Input
      , testDepends : List Input
      , doCheck : Bool
      , doHaddock : Bool
      , meta : { description : Text, license : Text }
      }

let mkHaskellPackage =
      \(name : Text) ->
      \(version : Text) ->
      \(src : Input) ->
        { name
        , version
        , src
        , isLibrary = True
        , isExecutable = False
        , buildDepends = [] : List Input
        , executableDepends = [] : List Input
        , testDepends = [] : List Input
        , doCheck = True
        , doHaddock = True
        , meta = { description = "", license = "" }
        } : HaskellDerivation

-- =============================================================================
-- Exports
-- =============================================================================

in  { System
    , OutputHashMode
    , OutputHashAlgo
    , Output
    , defaultOutput
    , Input
    , EnvVar
    , Derivation
    , emptyDerivation
    , FetchUrl
    , fetchurl
    , FetchGit
    , fetchgit
    , BuildPhase
    , MkDerivation
    , mkDerivation
    , RustDerivation
    , buildRustPackage
    , HaskellDerivation
    , mkHaskellPackage
    }
