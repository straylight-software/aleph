{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | Dhall emission for typed derivations.

This module converts Haskell Drv values to Dhall expressions that conform
to the schema in nix/prelude/dhall/derivation.dhall.

The Dhall output is validated by Nix against the store before building.
No shell strings are generated - aleph-exec reads the Dhall directly.

= Usage

@
import Aleph.Nix.Dhall (drvToDhall)
import Aleph.Nix.Derivation (Drv)

myPkg :: Drv
myPkg = ...

-- Emit Dhall expression
dhallExpr :: Text
dhallExpr = drvToDhall myPkg
@
-}
module Aleph.Nix.Dhall (
    drvToDhall,
    actionToDhall,
    srcToDhall,
    builderToDhall,
) where

import Data.Text (Text)
import qualified Data.Text as T

import Aleph.Nix.Derivation (
    Action (..),
    Builder (..),
    Deps (..),
    Drv (..),
    EnvVar (..),
    FetchGitHub (..),
    FetchUrl (..),
    Meta (..),
    Phases (..),
    Src (..),
    WrapAction (..),
 )
import Aleph.Nix.Types (SriHash (..), StorePath (..))

-- | Convert a Drv to a Dhall expression conforming to derivation.dhall
drvToDhall :: Drv -> Text
drvToDhall Drv{..} =
    T.unlines
        [ "let D = ./derivation.dhall"
        , ""
        , "in D.Derivation ::"
        , "  { pname = " <> quoted drvName
        , "  , version = " <> quoted drvVersion
        , "  , src = " <> srcToDhall drvSrc
        , "  , builder = " <> builderToDhall drvBuilder
        , "  , deps = " <> depsToDhall drvDeps
        , "  , phases = " <> phasesToDhall drvPhases
        , "  , meta = " <> metaToDhall drvMeta
        , "  , env = " <> envToDhall drvEnv
        , "  , patches = " <> listToDhall storePathToDhall drvPatches
        , "  , strictDeps = " <> boolToDhall drvStrictDeps
        , "  , doCheck = " <> boolToDhall drvDoCheck
        , "  , dontUnpack = " <> boolToDhall drvDontUnpack
        , "  }"
        ]

-- | Convert a source specification to Dhall
srcToDhall :: Src -> Text
srcToDhall (SrcGitHub FetchGitHub{..}) =
    "D.Src.GitHub { owner = "
        <> quoted ghOwner
        <> ", repo = "
        <> quoted ghRepo
        <> ", rev = "
        <> quoted ghRev
        <> ", hash = "
        <> quoted (unSriHash ghHash)
        <> " }"
srcToDhall (SrcUrl FetchUrl{..}) =
    "D.Src.Url { url = "
        <> quoted urlUrl
        <> ", hash = "
        <> quoted (unSriHash urlHash)
        <> " }"
srcToDhall (SrcStore sp) =
    "D.Src.Store " <> storePathToDhall sp
srcToDhall SrcNull =
    "D.Src.None"

-- | Convert builder to Dhall
builderToDhall :: Builder -> Text
builderToDhall (CMake flags buildType) =
    "D.Builder.CMake { flags = "
        <> listToDhall quoted flags
        <> ", buildType = "
        <> optionalToDhall quoted buildType
        <> " }"
builderToDhall (Autotools confFlags makeFlags) =
    "D.Builder.Autotools { configureFlags = "
        <> listToDhall quoted confFlags
        <> ", makeFlags = "
        <> listToDhall quoted makeFlags
        <> " }"
builderToDhall (Meson flags) =
    "D.Builder.Meson { flags = " <> listToDhall quoted flags <> " }"
builderToDhall (CustomBuilder buildPhase installPhase) =
    "D.Builder.Custom { buildPhase = "
        <> quoted buildPhase
        <> ", installPhase = "
        <> quoted installPhase
        <> " }"
builderToDhall NoBuilder =
    "D.Builder.None"

-- | Convert a single action to Dhall
actionToDhall :: Action -> Text
actionToDhall (WriteFile path content) =
    "D.Action.WriteFile { path = " <> quoted path <> ", content = " <> quoted content <> " }"
actionToDhall (Install mode src dst) =
    "D.Action.Install { mode = "
        <> T.pack (show mode)
        <> ", src = "
        <> quoted src
        <> ", dst = "
        <> quoted dst
        <> " }"
actionToDhall (Mkdir path) =
    "D.Action.Mkdir { path = " <> quoted path <> " }"
actionToDhall (Symlink target link) =
    "D.Action.Symlink { target = " <> quoted target <> ", link = " <> quoted link <> " }"
actionToDhall (Copy src dst) =
    "D.Action.Copy { from = " <> quoted src <> ", to = " <> quoted dst <> " }"
actionToDhall (Remove path) =
    "D.Action.Remove { path = " <> quoted path <> " }"
actionToDhall (Unzip dest) =
    "D.Action.Unzip { dest = " <> quoted dest <> " }"
actionToDhall (PatchElfRpath binary rpaths) =
    "D.Action.PatchElfRpath { binary = "
        <> quoted binary
        <> ", rpaths = "
        <> listToDhall (quoted . id) rpaths
        <> " }"
actionToDhall (PatchElfAddRpath binary rpaths) =
    "D.Action.PatchElfAddRpath { binary = "
        <> quoted binary
        <> ", rpaths = "
        <> listToDhall (quoted . id) rpaths
        <> " }"
actionToDhall (Substitute file replacements) =
    "D.Action.Substitute { file = "
        <> quoted file
        <> ", replacements = "
        <> listToDhall replacementToDhall replacements
        <> " }"
actionToDhall (Wrap program wrapActions) =
    "D.Action.Wrap { program = "
        <> quoted program
        <> ", actions = "
        <> listToDhall wrapActionToDhall wrapActions
        <> " }"
actionToDhall (Run cmd args) =
    -- Run is deprecated but we handle it for backward compat
    error $ "Run action is not supported in zero-bash mode: " <> T.unpack cmd
actionToDhall (ToolRun _ _) =
    -- ToolRun should be resolved to actual tool paths by now
    error "ToolRun should be resolved before Dhall emission"

-- | Convert replacement pair
replacementToDhall :: (Text, Text) -> Text
replacementToDhall (from, to) =
    "{ from = " <> quoted from <> ", to = " <> quoted to <> " }"

-- | Convert wrap action
wrapActionToDhall :: WrapAction -> Text
wrapActionToDhall (WrapPrefix var val) =
    "D.WrapAction.Prefix { var = " <> quoted var <> ", value = " <> quoted val <> " }"
wrapActionToDhall (WrapSuffix var val) =
    "D.WrapAction.Suffix { var = " <> quoted var <> ", value = " <> quoted val <> " }"
wrapActionToDhall (WrapSet var val) =
    "D.WrapAction.Set { var = " <> quoted var <> ", value = " <> quoted val <> " }"
wrapActionToDhall (WrapSetDefault var val) =
    "D.WrapAction.SetDefault { var = " <> quoted var <> ", value = " <> quoted val <> " }"
wrapActionToDhall (WrapUnset var) =
    "D.WrapAction.Unset { var = " <> quoted var <> " }"
wrapActionToDhall (WrapAddFlags flags) =
    "D.WrapAction.AddFlags { flags = " <> quoted flags <> " }"

-- | Convert dependencies
depsToDhall :: Deps -> Text
depsToDhall Deps{..} =
    T.unlines
        [ "  { nativeBuildInputs = " <> listToDhall quoted nativeBuildInputs
        , "  , buildInputs = " <> listToDhall quoted buildInputs
        , "  , propagatedBuildInputs = " <> listToDhall quoted propagatedBuildInputs
        , "  , checkInputs = " <> listToDhall quoted checkInputs
        , "  }"
        ]

-- | Convert phases
phasesToDhall :: Phases -> Text
phasesToDhall Phases{..} =
    T.unlines
        [ "  { postPatch = " <> listToDhall actionToDhall postPatch
        , "  , preConfigure = " <> listToDhall actionToDhall preConfigure
        , "  , installPhase = " <> listToDhall actionToDhall install
        , "  , postInstall = " <> listToDhall actionToDhall postInstall
        , "  , postFixup = " <> listToDhall actionToDhall postFixup
        , "  }"
        ]

-- | Convert metadata
metaToDhall :: Meta -> Text
metaToDhall Meta{..} =
    T.unlines
        [ "  { description = " <> quoted description
        , "  , homepage = " <> optionalToDhall quoted homepage
        , "  , license = " <> quoted license
        , "  , platforms = " <> listToDhall quoted platforms
        , "  , mainProgram = " <> optionalToDhall quoted mainProgram
        , "  }"
        ]

-- | Convert environment variables
envToDhall :: [EnvVar] -> Text
envToDhall = listToDhall envVarToDhall

envVarToDhall :: EnvVar -> Text
envVarToDhall EnvVar{..} =
    "{ name = " <> quoted envName <> ", value = " <> quoted envValue <> " }"

-- | Convert store path to Dhall
storePathToDhall :: StorePath -> Text
storePathToDhall (StorePath path) =
    -- For now, emit as a package reference that Nix will resolve
    "D.pkg " <> quoted path

--------------------------------------------------------------------------------
-- Dhall helpers
--------------------------------------------------------------------------------

-- | Quote a text value for Dhall
quoted :: Text -> Text
quoted t = "\"" <> escape t <> "\""

-- | Escape special characters in Dhall strings
escape :: Text -> Text
escape =
    T.replace "\\" "\\\\"
        . T.replace "\"" "\\\""
        . T.replace "\n" "\\n"
        . T.replace "\t" "\\t"
        . T.replace "${" "\\${"

-- | Convert list to Dhall
listToDhall :: (a -> Text) -> [a] -> Text
listToDhall _ [] = "[] : List _"
listToDhall f xs = "[ " <> T.intercalate ", " (map f xs) <> " ]"

-- | Convert optional to Dhall
optionalToDhall :: (a -> Text) -> Maybe a -> Text
optionalToDhall _ Nothing = "None Text"
optionalToDhall f (Just x) = "Some " <> f x

-- | Convert bool to Dhall
boolToDhall :: Bool -> Text
boolToDhall True = "True"
boolToDhall False = "False"

-- | Unwrap SriHash
unSriHash :: SriHash -> Text
unSriHash (SriHash h) = h
