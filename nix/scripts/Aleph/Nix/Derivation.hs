{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | Typed derivation construction for Nix WASM plugins.

This module provides Haskell types that map directly to Nix derivation
attributes, enabling compile-time validation of package definitions.

= Design

Following RFC-004, we build derivations with:

1. **Single fixpoint** - No nested mkDerivation/callPackage confusion
2. **Typed build phases** - Haskell functions, not bash strings
3. **Explicit deps** - Dependencies are named, not positional

= Example

@
zlibNg :: Drv
zlibNg = Drv
  { drvName = "zlib-ng"
  , drvVersion = "2.2.4"
  , drvSrc = FetchGitHub
      { owner = "zlib-ng"
      , repo = "zlib-ng"
      , rev = "2.2.4"
      , hash = "sha256-Khmrhp5qy4vvoQe4WgoogpjWrgcUB/q8zZeqIydthYg="
      }
  , drvBuilder = CMake
      { cmakeFlags =
          [ "-DBUILD_STATIC_LIBS=ON"
          , "-DBUILD_SHARED_LIBS=OFF"
          , "-DZLIB_COMPAT=ON"
          ]
      }
  , drvDeps = Deps
      { nativeBuildInputs = ["cmake", "pkg-config"]
      , buildInputs = ["gtest"]
      }
  , drvMeta = Meta
      { description = "zlib for next generation systems"
      , homepage = Just "https://github.com/zlib-ng/zlib-ng"
      , license = "zlib"
      }
  }
@
-}
module Aleph.Nix.Derivation (
    -- * Core types
    Drv (..),
    defaultDrv,

    -- * Source fetching
    Src (..),
    FetchGitHub (..),
    FetchUrl (..),

    -- * Build systems
    Builder (..),
    CMakeFlags,
    ConfigureFlags,
    MakeFlags,

    -- * Typed build phases
    Action (..),
    WrapAction (..),
    Phases (..),
    emptyPhases,

    -- * Dependencies
    Deps (..),
    emptyDeps,

    -- * Metadata
    Meta (..),
    emptyMeta,

    -- * Environment
    EnvVar (..),

    -- * Marshaling to Nix
    ToNixValue (..),
    drvToNixAttrs,

    -- * Re-exports for convenience
    StorePath (..),
    System (..),
    SriHash (..),
    sha256Hash,

    -- * Typed paths (re-exported from Types)
    OutPath (..),
    SrcPath (..),
    PkgRef (..),
    RpathEntry (..),
    outPath,
    srcPath,
    pkgRef,
    rpathOut,
    rpathPkg,
    rpathLit,
) where

import Data.Int (Int64)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T
import GHC.Generics (Generic)

import Aleph.Nix.Types (
    OutPath (..),
    PkgRef (..),
    RpathEntry (..),
    SrcPath (..),
    SriHash (..),
    StorePath (..),
    System (..),
    outPath,
    pkgRef,
    rpathLit,
    rpathOut,
    rpathPkg,
    sha256Hash,
    srcPath,
 )
import Aleph.Nix.Value

--------------------------------------------------------------------------------
-- Source types
--------------------------------------------------------------------------------

-- | Source specification for a derivation.
data Src
    = SrcGitHub !FetchGitHub
    | SrcUrl !FetchUrl
    | -- | Already in the store (for patched sources, etc.)
      SrcStore !StorePath
    | -- | No source (meta-packages)
      SrcNull
    deriving (Eq, Show, Generic)

-- | Fetch from GitHub.
data FetchGitHub = FetchGitHub
    { ghOwner :: !Text
    , ghRepo :: !Text
    , ghRev :: !Text
    , ghHash :: !SriHash
    -- ^ Content hash of the fetched source
    }
    deriving (Eq, Show, Generic)

-- | Fetch from URL.
data FetchUrl = FetchUrl
    { urlUrl :: !Text
    , urlHash :: !SriHash
    -- ^ Content hash of the fetched file
    }
    deriving (Eq, Show, Generic)

--------------------------------------------------------------------------------
-- Build system types
--------------------------------------------------------------------------------

-- | Build system specification.
data Builder
    = CMake
        { cmakeFlags :: !CMakeFlags
        , cmakeBuildType :: !(Maybe Text)
        -- ^ "Release", "Debug", etc.
        }
    | Autotools
        { configureFlags :: !ConfigureFlags
        , makeFlags :: !MakeFlags
        }
    | Meson
        { mesonFlags :: ![Text]
        }
    | CustomBuilder
        { buildPhase :: !Text
        -- ^ Shell commands for build
        , installPhase :: !Text
        -- ^ Shell commands for install
        }
    | -- | Header-only, meta-package, etc.
      NoBuilder
    deriving (Eq, Show, Generic)

type CMakeFlags = [Text]
type ConfigureFlags = [Text]
type MakeFlags = [Text]

--------------------------------------------------------------------------------
-- Typed build phases
--------------------------------------------------------------------------------

{- | Typed build action.

These replace shell strings with structured, type-safe operations.
The Nix-side builder translates these to actual shell commands,
but the Haskell side never deals with quoting, escaping, or string interpolation.

= CA Soundness

Every action is deterministic:
- File contents are explicit Text values
- Paths are structured, not interpolated strings
- No shell expansion or globbing
-}
data Action
    = {- | Write a file with given contents.
      @WriteFile "include/mdspan" content@ → writes content to $out/include/mdspan
      -}
      WriteFile !Text !Text
    | {- | Install a file with mode.
      @Install 0o644 "src/foo.h" "include/foo.h"@ → install -m644 src → $out/include
      -}
      Install !Int !Text !Text
    | -- | Create directory (mkdir -p).
      Mkdir !Text
    | {- | Create symlink.
      @Symlink target linkName@
      -}
      Symlink !Text !Text
    | {- | Copy file or directory.
      @Copy src dst@
      -}
      Copy !Text !Text
    | -- | Remove file or directory.
      Remove !Text
    | {- | Unzip source archive to a directory.
      @Unzip "unpacked"@ → unzip $src -d unpacked
      Used for wheel extraction where dontUnpack=true.
      -}
      Unzip !Text
    | {- | Set rpath on an ELF binary.
      @PatchElfRpath "bin/foo" ["libdir", "lib"]@ → patchelf --set-rpath ... bin/foo
      Paths are relative to $out unless they start with /.
      -}
      PatchElfRpath !Text ![Text]
    | {- | Add to rpath on an ELF binary.
      @PatchElfAddRpath "lib/libfoo.so" ["@libgcc@/lib"]@
      Placeholders like @foo@ are resolved to package paths.
      -}
      PatchElfAddRpath !Text ![Text]
    | {- | Replace string in file.
      @Substitute "src/config.h" [("@PREFIX@", "$out"), ("@VERSION@", "1.0")]@
      Typed substituteInPlace - no shell escaping bugs.
      -}
      Substitute !Text ![(Text, Text)]
    | {- | Wrap a program with environment setup.
      @Wrap "bin/mytool" [WrapPrefix "PATH" paths, WrapSet "SSL_CERT_FILE" cert]@
      -}
      Wrap !Text ![WrapAction]
    | {- | Run arbitrary command (escape hatch, avoid if possible).

      WARNING: This bypasses type safety. Prefer typed actions.
      If you must use this, the tool will be added to nativeBuildInputs
      but arguments are unchecked.
      -}
      Run !Text ![Text]
    | {- | Typed tool invocation with automatic dependency tracking.
      The PkgRef is added to nativeBuildInputs automatically.
      -}
      ToolRun !PkgRef ![Text]
    deriving (Eq, Show, Generic)

-- | Wrapper actions for the Wrap action.
data WrapAction
    = -- | Prefix a variable: --prefix VAR : value
      WrapPrefix !Text !Text
    | -- | Suffix a variable: --suffix VAR : value
      WrapSuffix !Text !Text
    | -- | Set a variable: --set VAR value
      WrapSet !Text !Text
    | -- | Set default (only if unset): --set-default VAR value
      WrapSetDefault !Text !Text
    | -- | Unset a variable: --unset VAR
      WrapUnset !Text
    | -- | Add flags: --add-flags "flags"
      WrapAddFlags !Text
    deriving (Eq, Show, Generic)

{- | Typed build phases.

Each phase is a list of actions executed in order.
Empty list = phase not customized (uses builder defaults).
-}
data Phases = Phases
    { postPatch :: ![Action]
    -- ^ After patches applied, before configure
    , preConfigure :: ![Action]
    -- ^ Before configure phase
    , install :: ![Action]
    -- ^ Custom install phase (replaces default)
    , postInstall :: ![Action]
    -- ^ After install phase (most common customization)
    , postFixup :: ![Action]
    -- ^ After fixup phase (wrapping, patching rpaths)
    }
    deriving (Eq, Show, Generic)

-- | Empty phases (no customization).
emptyPhases :: Phases
emptyPhases = Phases [] [] [] [] []

--------------------------------------------------------------------------------
-- Dependencies
--------------------------------------------------------------------------------

{- | Package dependencies.

Dependencies are specified by name (string), resolved by Nix at eval time.
This keeps the Haskell side pure - no package set threading.
-}
data Deps = Deps
    { nativeBuildInputs :: ![Text]
    -- ^ Build-time tools (cmake, pkg-config)
    , buildInputs :: ![Text]
    -- ^ Libraries to link against
    , propagatedBuildInputs :: ![Text]
    -- ^ Transitive deps
    , checkInputs :: ![Text]
    -- ^ Test dependencies
    }
    deriving (Eq, Show, Generic)

emptyDeps :: Deps
emptyDeps = Deps [] [] [] []

--------------------------------------------------------------------------------
-- Metadata
--------------------------------------------------------------------------------

-- | Package metadata.
data Meta = Meta
    { description :: !Text
    , homepage :: !(Maybe Text)
    , license :: !Text
    -- ^ License identifier (e.g., "mit", "gpl3", "zlib")
    , platforms :: ![Text]
    -- ^ Empty = all platforms
    , mainProgram :: !(Maybe Text)
    -- ^ For packages providing a binary
    }
    deriving (Eq, Show, Generic)

emptyMeta :: Meta
emptyMeta = Meta "" Nothing "" [] Nothing

--------------------------------------------------------------------------------
-- Environment variables
--------------------------------------------------------------------------------

-- | Environment variable to set during build.
data EnvVar = EnvVar
    { envName :: !Text
    , envValue :: !Text
    }
    deriving (Eq, Show, Generic)

--------------------------------------------------------------------------------
-- The derivation type
--------------------------------------------------------------------------------

{- | A complete derivation specification.

This is the single source of truth for a package. No nested fixpoints,
no override/overrideAttrs confusion.

= CA Soundness

Every field that affects the build output is explicit:

- @drvSrc@ has a content hash (SriHash)
- @drvPatches@ are StorePaths (content-addressed)
- @drvDeps@ are package names resolved deterministically
- @drvBuilder@ fully specifies the build system config

With CA derivations, the output path is determined by the output content,
not by this specification. But this specification determines *what* gets
built, so it must be complete.
-}
data Drv = Drv
    { drvName :: !Text
    , drvVersion :: !Text
    , drvSrc :: !Src
    , drvBuilder :: !Builder
    , drvDeps :: !Deps
    , drvMeta :: !Meta
    , drvEnv :: ![EnvVar]
    -- ^ Extra environment variables
    , drvPatches :: ![StorePath]
    -- ^ Patch files (must be in store)
    , drvPhases :: !Phases
    -- ^ Typed build phase customizations
    , drvStrictDeps :: !Bool
    , drvDoCheck :: !Bool
    , drvDontUnpack :: !Bool
    -- ^ Skip unpack phase (for wheels, pre-extracted sources)
    , drvSystem :: !(Maybe System)
    -- ^ Target system (Nothing = current)
    }
    deriving (Eq, Show, Generic)

-- | Default derivation with sensible defaults.
defaultDrv :: Drv
defaultDrv =
    Drv
        { drvName = ""
        , drvVersion = ""
        , drvSrc = SrcNull
        , drvBuilder = NoBuilder
        , drvDeps = emptyDeps
        , drvMeta = emptyMeta
        , drvEnv = []
        , drvPatches = []
        , drvPhases = emptyPhases
        , drvStrictDeps = True
        , drvDoCheck = False
        , drvDontUnpack = False
        , drvSystem = Nothing -- inherit from builder
        }

--------------------------------------------------------------------------------
-- ToNixValue typeclass
--------------------------------------------------------------------------------

-- | Convert Haskell values to Nix Values.
class ToNixValue a where
    toNixValue :: a -> IO Value

instance ToNixValue Bool where
    toNixValue = mkBool

instance ToNixValue Int64 where
    toNixValue = mkInt

instance ToNixValue Text where
    toNixValue = mkString

instance (ToNixValue a) => ToNixValue [a] where
    toNixValue xs = do
        vals <- mapM toNixValue xs
        mkList vals

instance (ToNixValue a) => ToNixValue (Maybe a) where
    toNixValue Nothing = mkNull
    toNixValue (Just x) = toNixValue x

instance (ToNixValue a) => ToNixValue (Map Text a) where
    toNixValue m = do
        pairs <- mapM (\(k, v) -> (k,) <$> toNixValue v) (Map.toList m)
        mkAttrs (Map.fromList pairs)

instance ToNixValue StorePath where
    toNixValue (StorePath p) = mkPath p

instance ToNixValue System where
    toNixValue (System s) = mkString s

instance ToNixValue SriHash where
    toNixValue (SriHash h) = mkString h

--------------------------------------------------------------------------------
-- Derivation to Nix attrset
--------------------------------------------------------------------------------

{- | Convert a Drv to a Nix attrset suitable for the derivation builder.

This produces an attrset that can be consumed by a Nix-side function
to produce the actual derivation. The Nix side handles:

- Resolving dependency names to actual packages
- Calling fetchFromGitHub/fetchurl
- Setting up the stdenv builder

The returned attrset has this shape:

@
{
  pname = "zlib-ng";
  version = "2.2.4";
  src = { type = "github"; owner = "zlib-ng"; ... };
  builder = { type = "cmake"; flags = [...]; };
  deps = { nativeBuildInputs = [...]; buildInputs = [...]; };
  meta = { description = "..."; license = "zlib"; };
  ...
}
@
-}
drvToNixAttrs :: Drv -> IO Value
drvToNixAttrs Drv{..} = do
    -- Build the attrset incrementally
    pairs <-
        sequence
            [ ("pname",) <$> mkString drvName
            , ("version",) <$> mkString drvVersion
            , ("src",) <$> srcToNix drvSrc
            , ("builder",) <$> builderToNix drvBuilder
            , ("deps",) <$> depsToNix drvDeps
            , ("meta",) <$> metaToNix drvMeta
            , ("env",) <$> envToNix drvEnv
            , ("patches",) <$> toNixValue drvPatches
            , ("phases",) <$> phasesToNix drvPhases
            , ("strictDeps",) <$> mkBool drvStrictDeps
            , ("doCheck",) <$> mkBool drvDoCheck
            , ("dontUnpack",) <$> mkBool drvDontUnpack
            , ("system",) <$> toNixValue drvSystem
            ]
    mkAttrs (Map.fromList pairs)

srcToNix :: Src -> IO Value
srcToNix (SrcGitHub FetchGitHub{..}) = do
    pairs <-
        sequence
            [ ("type",) <$> mkString "github"
            , ("owner",) <$> mkString ghOwner
            , ("repo",) <$> mkString ghRepo
            , ("rev",) <$> mkString ghRev
            , ("hash",) <$> toNixValue ghHash
            ]
    mkAttrs (Map.fromList pairs)
srcToNix (SrcUrl FetchUrl{..}) = do
    pairs <-
        sequence
            [ ("type",) <$> mkString "url"
            , ("url",) <$> mkString urlUrl
            , ("hash",) <$> toNixValue urlHash
            ]
    mkAttrs (Map.fromList pairs)
srcToNix (SrcStore p) = do
    pairs <-
        sequence
            [ ("type",) <$> mkString "store"
            , ("path",) <$> toNixValue p
            ]
    mkAttrs (Map.fromList pairs)
srcToNix SrcNull = mkNull

builderToNix :: Builder -> IO Value
builderToNix (CMake flags buildType) = do
    flagsVal <- toNixValue flags
    buildTypeVal <- toNixValue buildType
    pairs <-
        sequence
            [ pure ("type",) <*> mkString "cmake"
            , pure ("flags", flagsVal)
            , pure ("buildType", buildTypeVal)
            ]
    mkAttrs (Map.fromList pairs)
builderToNix (Autotools confFlags makeFlags) = do
    confVal <- toNixValue confFlags
    makeVal <- toNixValue makeFlags
    pairs <-
        sequence
            [ pure ("type",) <*> mkString "autotools"
            , pure ("configureFlags", confVal)
            , pure ("makeFlags", makeVal)
            ]
    mkAttrs (Map.fromList pairs)
builderToNix (Meson flags) = do
    flagsVal <- toNixValue flags
    pairs <-
        sequence
            [ pure ("type",) <*> mkString "meson"
            , pure ("flags", flagsVal)
            ]
    mkAttrs (Map.fromList pairs)
builderToNix (CustomBuilder build install) = do
    pairs <-
        sequence
            [ ("type",) <$> mkString "custom"
            , ("buildPhase",) <$> mkString build
            , ("installPhase",) <$> mkString install
            ]
    mkAttrs (Map.fromList pairs)
builderToNix NoBuilder = do
    pairs <- sequence [("type",) <$> mkString "none"]
    mkAttrs (Map.fromList pairs)

depsToNix :: Deps -> IO Value
depsToNix Deps{..} = do
    pairs <-
        sequence
            [ ("nativeBuildInputs",) <$> toNixValue nativeBuildInputs
            , ("buildInputs",) <$> toNixValue buildInputs
            , ("propagatedBuildInputs",) <$> toNixValue propagatedBuildInputs
            , ("checkInputs",) <$> toNixValue checkInputs
            ]
    mkAttrs (Map.fromList pairs)

metaToNix :: Meta -> IO Value
metaToNix Meta{..} = do
    pairs <-
        sequence
            [ ("description",) <$> mkString description
            , ("homepage",) <$> toNixValue homepage
            , ("license",) <$> mkString license
            , ("platforms",) <$> toNixValue platforms
            , ("mainProgram",) <$> toNixValue mainProgram
            ]
    mkAttrs (Map.fromList pairs)

envToNix :: [EnvVar] -> IO Value
envToNix vars = do
    pairs <- mapM (\EnvVar{..} -> (envName,) <$> mkString envValue) vars
    mkAttrs (Map.fromList pairs)

--------------------------------------------------------------------------------
-- Phases serialization
--------------------------------------------------------------------------------

phasesToNix :: Phases -> IO Value
phasesToNix Phases{..} = do
    pairs <-
        sequence
            [ ("postPatch",) <$> actionsToNix postPatch
            , ("preConfigure",) <$> actionsToNix preConfigure
            , ("installPhase",) <$> actionsToNix install
            , ("postInstall",) <$> actionsToNix postInstall
            , ("postFixup",) <$> actionsToNix postFixup
            ]
    mkAttrs (Map.fromList pairs)

actionsToNix :: [Action] -> IO Value
actionsToNix actions = do
    vals <- mapM actionToNix actions
    mkList vals

actionToNix :: Action -> IO Value
actionToNix (WriteFile path content) = do
    pairs <-
        sequence
            [ ("action",) <$> mkString "writeFile"
            , ("path",) <$> mkString path
            , ("content",) <$> mkString content
            ]
    mkAttrs (Map.fromList pairs)
actionToNix (Install mode src dst) = do
    pairs <-
        sequence
            [ ("action",) <$> mkString "install"
            , ("mode",) <$> mkInt (fromIntegral mode)
            , ("src",) <$> mkString src
            , ("dst",) <$> mkString dst
            ]
    mkAttrs (Map.fromList pairs)
actionToNix (Mkdir path) = do
    pairs <-
        sequence
            [ ("action",) <$> mkString "mkdir"
            , ("path",) <$> mkString path
            ]
    mkAttrs (Map.fromList pairs)
actionToNix (Symlink target linkName) = do
    pairs <-
        sequence
            [ ("action",) <$> mkString "symlink"
            , ("target",) <$> mkString target
            , ("link",) <$> mkString linkName
            ]
    mkAttrs (Map.fromList pairs)
actionToNix (Copy src dst) = do
    pairs <-
        sequence
            [ ("action",) <$> mkString "copy"
            , ("src",) <$> mkString src
            , ("dst",) <$> mkString dst
            ]
    mkAttrs (Map.fromList pairs)
actionToNix (Remove path) = do
    pairs <-
        sequence
            [ ("action",) <$> mkString "remove"
            , ("path",) <$> mkString path
            ]
    mkAttrs (Map.fromList pairs)
actionToNix (Unzip destDir) = do
    pairs <-
        sequence
            [ ("action",) <$> mkString "unzip"
            , ("dest",) <$> mkString destDir
            ]
    mkAttrs (Map.fromList pairs)
actionToNix (PatchElfRpath path rpaths) = do
    rpathsVal <- toNixValue rpaths
    pairs <-
        sequence
            [ ("action",) <$> mkString "patchelfRpath"
            , ("path",) <$> mkString path
            , pure ("rpaths", rpathsVal)
            ]
    mkAttrs (Map.fromList pairs)
actionToNix (PatchElfAddRpath path rpaths) = do
    rpathsVal <- toNixValue rpaths
    pairs <-
        sequence
            [ ("action",) <$> mkString "patchelfAddRpath"
            , ("path",) <$> mkString path
            , pure ("rpaths", rpathsVal)
            ]
    mkAttrs (Map.fromList pairs)
actionToNix (Substitute file replacements) = do
    -- Convert [(Text, Text)] to list of {from, to} attrsets
    reps <-
        mapM
            ( \(from, to) -> do
                fromVal <- mkString from
                toVal <- mkString to
                mkAttrs (Map.fromList [("from", fromVal), ("to", toVal)])
            )
            replacements
    repsVal <- mkList reps
    pairs <-
        sequence
            [ ("action",) <$> mkString "substitute"
            , ("file",) <$> mkString file
            , pure ("replacements", repsVal)
            ]
    mkAttrs (Map.fromList pairs)
actionToNix (Wrap program wrapActions) = do
    actionsVal <- mapM wrapActionToNix wrapActions
    actionsListVal <- mkList actionsVal
    pairs <-
        sequence
            [ ("action",) <$> mkString "wrap"
            , ("program",) <$> mkString program
            , pure ("wrapActions", actionsListVal)
            ]
    mkAttrs (Map.fromList pairs)
actionToNix (Run cmd args) = do
    argsVal <- toNixValue args
    pairs <-
        sequence
            [ ("action",) <$> mkString "run"
            , ("cmd",) <$> mkString cmd
            , pure ("args", argsVal)
            ]
    mkAttrs (Map.fromList pairs)
actionToNix (ToolRun (PkgRef pkg) args) = do
    argsVal <- toNixValue args
    pairs <-
        sequence
            [ ("action",) <$> mkString "toolRun"
            , ("pkg",) <$> mkString pkg
            , pure ("args", argsVal)
            ]
    mkAttrs (Map.fromList pairs)

wrapActionToNix :: WrapAction -> IO Value
wrapActionToNix (WrapPrefix var val) = do
    pairs <-
        sequence
            [ ("type",) <$> mkString "prefix"
            , ("var",) <$> mkString var
            , ("value",) <$> mkString val
            ]
    mkAttrs (Map.fromList pairs)
wrapActionToNix (WrapSuffix var val) = do
    pairs <-
        sequence
            [ ("type",) <$> mkString "suffix"
            , ("var",) <$> mkString var
            , ("value",) <$> mkString val
            ]
    mkAttrs (Map.fromList pairs)
wrapActionToNix (WrapSet var val) = do
    pairs <-
        sequence
            [ ("type",) <$> mkString "set"
            , ("var",) <$> mkString var
            , ("value",) <$> mkString val
            ]
    mkAttrs (Map.fromList pairs)
wrapActionToNix (WrapSetDefault var val) = do
    pairs <-
        sequence
            [ ("type",) <$> mkString "setDefault"
            , ("var",) <$> mkString var
            , ("value",) <$> mkString val
            ]
    mkAttrs (Map.fromList pairs)
wrapActionToNix (WrapUnset var) = do
    pairs <-
        sequence
            [ ("type",) <$> mkString "unset"
            , ("var",) <$> mkString var
            ]
    mkAttrs (Map.fromList pairs)
wrapActionToNix (WrapAddFlags flags) = do
    pairs <-
        sequence
            [ ("type",) <$> mkString "addFlags"
            , ("flags",) <$> mkString flags
            ]
    mkAttrs (Map.fromList pairs)
