{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}

{- | Nix-like syntax for package definitions.

This module provides a familiar API for nixers. The goal is minimal
syntax distance from .nix files.

= Comparison

Nix:
@
stdenv.mkDerivation {
  pname = "rapidjson";
  version = "1.1.0";

  src = fetchFromGitHub {
    owner = "Tencent";
    repo = "rapidjson";
    rev = "v1.1.0";
    hash = "sha256-...";
  };

  nativeBuildInputs = [ cmake doxygen ];
  buildInputs = [ gtest ];
  cmakeFlags = [ "-DRAPIDJSON_BUILD_DOC=ON" ];

  postPatch = \'\'
    substituteInPlace doc/Doxyfile --replace-fail "YES" "NO"
  \'\';
}
@

Haskell:
@
rapidjson = mkDerivation
  [ pname   "rapidjson"
  , version "1.1.0"

  , src $ fetchFromGitHub
      [ owner "Tencent"
      , repo  "rapidjson"
      , rev   "v1.1.0"
      , hash  "sha256-..."
      ]

  , nativeBuildInputs ["cmake", "doxygen"]
  , buildInputs       ["gtest"]
  , cmakeFlags        ["-DRAPIDJSON_BUILD_DOC=ON"]

  , postPatch
      [ substitute "doc/Doxyfile" [("YES", "NO")]
      ]
  ]
@
-}
module Aleph.Nix.Syntax (
    -- * Package definition
    mkDerivation,
    Attr,

    -- * Basic attributes
    pname,
    version,
    src,
    strictDeps,
    doCheck,
    dontUnpack,

    -- * Source fetching
    fetchFromGitHub,
    fetchurl,
    SrcAttr,
    owner,
    repo,
    rev,
    hash,
    url,

    -- * Dependencies
    nativeBuildInputs,
    buildInputs,
    propagatedBuildInputs,
    checkInputs,
    patches,

    -- * Build systems (typed)
    cmake,

    -- * Build system flags (raw, prefer typed versions above)
    cmakeFlags,
    configureFlags,
    makeFlags,
    mesonFlags,

    -- * Phases (typed actions)
    postPatch,
    preConfigure,
    installPhase,
    postInstall,
    postFixup,

    -- * Phase actions
    writeFile,
    substitute,
    mkdir,
    copy,
    symlink,
    remove,
    unzip,
    patchElfRpath,
    patchElfAddRpath,
    wrap,
    wrapPrefix,
    wrapSuffix,
    wrapSet,
    wrapSetDefault,
    wrapUnset,
    wrapAddFlags,
    run,
    tool,

    -- * Metadata
    description,
    homepage,
    license,
    mainProgram,

    -- * Re-export CMake options for convenience
    module Aleph.Script.Tools.CMake,
) where

import Aleph.Nix.Derivation (
    Action (..),
    Builder (Autotools, CMake, CustomBuilder, Meson, NoBuilder),
    Deps,
    Drv (..),
    FetchGitHub (..),
    FetchUrl (..),
    Meta,
    Phases,
    Src (..),
    WrapAction (..),
    defaultDrv,
    emptyDeps,
    emptyMeta,
    emptyPhases,
    sha256Hash,
 )
import qualified Aleph.Nix.Derivation as D
import Aleph.Nix.Types (PkgRef (..), StorePath (..))
import Aleph.Script.Tools.CMake (BuildType (..), Options (..), buildArgs, defaults)
import qualified Aleph.Script.Tools.CMake
import Data.Text (Text)
import qualified Data.Text as T
import Prelude hiding (unzip, writeFile)

--------------------------------------------------------------------------------
-- Attribute accumulator
--------------------------------------------------------------------------------

data Attr
    = PnameA Text
    | VersionA Text
    | SrcA Src
    | StrictDepsA Bool
    | DoCheckA Bool
    | DontUnpackA Bool
    | NativeBuildInputsA [Text]
    | BuildInputsA [Text]
    | PropagatedBuildInputsA [Text]
    | CheckInputsA [Text]
    | PatchesA [StorePath]
    | CMakeA Aleph.Script.Tools.CMake.Options -- typed cmake
    | CmakeFlagsA [Text] -- raw flags (escape hatch)
    | ConfigureFlagsA [Text]
    | MakeFlagsA [Text]
    | MesonFlagsA [Text]
    | PostPatchA [Action]
    | PreConfigureA [Action]
    | InstallPhaseA [Action]
    | PostInstallA [Action]
    | PostFixupA [Action]
    | DescriptionA Text
    | HomepageA Text
    | LicenseA Text
    | MainProgramA Text

--------------------------------------------------------------------------------
-- Source fetching
--------------------------------------------------------------------------------

data SrcAttr
    = OwnerA Text
    | RepoA Text
    | RevA Text
    | HashA Text
    | UrlA Text

owner, repo, rev, hash, url :: Text -> SrcAttr
owner = OwnerA
repo = RepoA
rev = RevA
hash = HashA
url = UrlA

fetchFromGitHub :: [SrcAttr] -> Src
fetchFromGitHub attrs =
    SrcGitHub
        FetchGitHub
            { ghOwner = find isOwner ""
            , ghRepo = find isRepo ""
            , ghRev = find isRev ""
            , ghHash = sha256Hash $ stripSha $ find isHash ""
            }
  where
    find p def = case filter p attrs of
        (a : _) -> extract a
        [] -> def

    isOwner (OwnerA _) = True; isOwner _ = False
    isRepo (RepoA _) = True; isRepo _ = False
    isRev (RevA _) = True; isRev _ = False
    isHash (HashA _) = True; isHash _ = False

    extract (OwnerA t) = t
    extract (RepoA t) = t
    extract (RevA t) = t
    extract (HashA t) = t
    extract (UrlA t) = t

    stripSha t = maybe t id $ T.stripPrefix "sha256-" t

fetchurl :: [SrcAttr] -> Src
fetchurl attrs =
    SrcUrl
        FetchUrl
            { urlUrl = find isUrl ""
            , urlHash = sha256Hash $ stripSha $ find isHash ""
            }
  where
    find p def = case filter p attrs of
        (a : _) -> extract a
        [] -> def

    isUrl (UrlA _) = True; isUrl _ = False
    isHash (HashA _) = True; isHash _ = False

    extract (UrlA t) = t
    extract (HashA t) = t
    extract _ = ""

    stripSha t = maybe t id $ T.stripPrefix "sha256-" t

--------------------------------------------------------------------------------
-- Basic attributes
--------------------------------------------------------------------------------

pname :: Text -> Attr
pname = PnameA

version :: Text -> Attr
version = VersionA

src :: Src -> Attr
src = SrcA

strictDeps :: Bool -> Attr
strictDeps = StrictDepsA

doCheck :: Bool -> Attr
doCheck = DoCheckA

dontUnpack :: Bool -> Attr
dontUnpack = DontUnpackA

--------------------------------------------------------------------------------
-- Dependencies
--------------------------------------------------------------------------------

nativeBuildInputs :: [Text] -> Attr
nativeBuildInputs = NativeBuildInputsA

buildInputs :: [Text] -> Attr
buildInputs = BuildInputsA

propagatedBuildInputs :: [Text] -> Attr
propagatedBuildInputs = PropagatedBuildInputsA

checkInputs :: [Text] -> Attr
checkInputs = CheckInputsA

patches :: [StorePath] -> Attr
patches = PatchesA

--------------------------------------------------------------------------------
-- Build systems (typed)
--------------------------------------------------------------------------------

{- | Typed CMake configuration. Generates flags from structured options.

@
cmake defaults
    { buildStaticLibs = Just True
    , buildSharedLibs = Just False
    , buildType = Just Release
    }
@

Equivalent to:
@
cmakeFlags ["-DBUILD_STATIC_LIBS=ON", "-DBUILD_SHARED_LIBS=OFF", "-DCMAKE_BUILD_TYPE=Release"]
@
-}
cmake :: Aleph.Script.Tools.CMake.Options -> Attr
cmake = CMakeA

--------------------------------------------------------------------------------
-- Build flags (raw - prefer typed versions above)
--------------------------------------------------------------------------------

cmakeFlags :: [Text] -> Attr
cmakeFlags = CmakeFlagsA

configureFlags :: [Text] -> Attr
configureFlags = ConfigureFlagsA

makeFlags :: [Text] -> Attr
makeFlags = MakeFlagsA

mesonFlags :: [Text] -> Attr
mesonFlags = MesonFlagsA

--------------------------------------------------------------------------------
-- Phase actions
--------------------------------------------------------------------------------

postPatch :: [Action] -> Attr
postPatch = PostPatchA

preConfigure :: [Action] -> Attr
preConfigure = PreConfigureA

installPhase :: [Action] -> Attr
installPhase = InstallPhaseA

postInstall :: [Action] -> Attr
postInstall = PostInstallA

postFixup :: [Action] -> Attr
postFixup = PostFixupA

-- | Write content to a file (relative to $out)
writeFile :: Text -> Text -> Action
writeFile = WriteFile

-- | substituteInPlace with typed replacement pairs
substitute :: Text -> [(Text, Text)] -> Action
substitute = Substitute

-- | mkdir -p (relative to $out)
mkdir :: Text -> Action
mkdir = Mkdir

-- | cp -r src dst (relative to $out)
copy :: Text -> Text -> Action
copy = Copy

-- | ln -s target link (relative to $out)
symlink :: Text -> Text -> Action
symlink = Symlink

-- | rm -rf (relative to $out)
remove :: Text -> Action
remove = Remove

-- | Unzip $src to a directory (for wheel extraction with dontUnpack=true)
unzip :: Text -> Action
unzip = Unzip

-- | Set rpath on an ELF binary (paths relative to $out unless absolute)
patchElfRpath :: Text -> [Text] -> Action
patchElfRpath = PatchElfRpath

-- | Add to rpath on an ELF binary
patchElfAddRpath :: Text -> [Text] -> Action
patchElfAddRpath = PatchElfAddRpath

-- | Wrap a program with environment modifications
wrap :: Text -> [WrapAction] -> Action
wrap = Wrap

-- | Prefix a PATH-like variable
wrapPrefix :: Text -> Text -> WrapAction
wrapPrefix = WrapPrefix

-- | Suffix a PATH-like variable
wrapSuffix :: Text -> Text -> WrapAction
wrapSuffix = WrapSuffix

-- | Set an environment variable
wrapSet :: Text -> Text -> WrapAction
wrapSet = WrapSet

-- | Set default value (only if unset)
wrapSetDefault :: Text -> Text -> WrapAction
wrapSetDefault = WrapSetDefault

-- | Unset an environment variable
wrapUnset :: Text -> WrapAction
wrapUnset = WrapUnset

-- | Add flags to the program
wrapAddFlags :: Text -> WrapAction
wrapAddFlags = WrapAddFlags

-- | Run arbitrary command (escape hatch - prefer 'tool' for tracked deps)
run :: Text -> [Text] -> Action
run = Run

{- | Typed tool invocation with automatic dependency tracking.

The package is automatically added to nativeBuildInputs.

@
postInstall
    [ tool "jq" ["-r", ".name", "package.json"]
    , tool "patchelf" ["--set-rpath", "$out/lib", "$out/bin/foo"]
    ]
@

Prefer this over 'run' - the tool dependency is tracked and validated.
-}
tool :: Text -> [Text] -> Action
tool pkg args = ToolRun (PkgRef pkg) args

--------------------------------------------------------------------------------
-- Metadata
--------------------------------------------------------------------------------

{- | NOTE: meta combinator is deprecated - use description, homepage, etc. directly
meta :: [Attr] -> Attr
meta attrs = ...  -- Would need safe head or NonEmpty
-}
description :: Text -> Attr
description = DescriptionA

homepage :: Text -> Attr
homepage = HomepageA

license :: Text -> Attr
license = LicenseA

mainProgram :: Text -> Attr
mainProgram = MainProgramA

--------------------------------------------------------------------------------
-- mkDerivation
--------------------------------------------------------------------------------

mkDerivation :: [Attr] -> Drv
mkDerivation = foldr apply defaultDrv
  where
    apply :: Attr -> Drv -> Drv
    apply (PnameA t) d = d{drvName = t}
    apply (VersionA t) d = d{drvVersion = t}
    apply (SrcA s) d = d{drvSrc = s}
    apply (StrictDepsA b) d = d{drvStrictDeps = b}
    apply (DoCheckA b) d = d{drvDoCheck = b}
    apply (DontUnpackA b) d = d{drvDontUnpack = b}
    apply (NativeBuildInputsA xs) d =
        d{drvDeps = (drvDeps d){D.nativeBuildInputs = xs}}
    apply (BuildInputsA xs) d =
        d{drvDeps = (drvDeps d){D.buildInputs = xs}}
    apply (PropagatedBuildInputsA xs) d =
        d{drvDeps = (drvDeps d){D.propagatedBuildInputs = xs}}
    apply (CheckInputsA xs) d =
        d{drvDeps = (drvDeps d){D.checkInputs = xs}}
    apply (PatchesA ps) d = d{drvPatches = ps}
    apply (CMakeA opts) d =
        d{drvBuilder = CMake{D.cmakeFlags = buildArgs opts, D.cmakeBuildType = Nothing}}
    apply (CmakeFlagsA fs) d =
        d{drvBuilder = CMake{D.cmakeFlags = fs, D.cmakeBuildType = Nothing}}
    apply (ConfigureFlagsA fs) d = case drvBuilder d of
        Autotools _ mf -> d{drvBuilder = Autotools fs mf}
        _ -> d{drvBuilder = Autotools fs []}
    apply (MakeFlagsA fs) d = case drvBuilder d of
        Autotools cf _ -> d{drvBuilder = Autotools cf fs}
        _ -> d{drvBuilder = Autotools [] fs}
    apply (MesonFlagsA fs) d = d{drvBuilder = Meson fs}
    apply (PostPatchA as) d =
        d{drvPhases = (drvPhases d){D.postPatch = as}}
    apply (PreConfigureA as) d =
        d{drvPhases = (drvPhases d){D.preConfigure = as}}
    apply (InstallPhaseA as) d =
        d{drvPhases = (drvPhases d){D.install = as}}
    apply (PostInstallA as) d =
        d{drvPhases = (drvPhases d){D.postInstall = as}}
    apply (PostFixupA as) d =
        d{drvPhases = (drvPhases d){D.postFixup = as}}
    apply (DescriptionA t) d =
        d{drvMeta = (drvMeta d){D.description = t}}
    apply (HomepageA t) d =
        d{drvMeta = (drvMeta d){D.homepage = Just t}}
    apply (LicenseA t) d =
        d{drvMeta = (drvMeta d){D.license = t}}
    apply (MainProgramA t) d =
        d{drvMeta = (drvMeta d){D.mainProgram = Just t}}
