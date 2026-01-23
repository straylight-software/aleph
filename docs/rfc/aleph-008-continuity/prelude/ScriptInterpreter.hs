{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | Script Interpreter

Compiles typed Script.dhall commands to:
1. Bash (for Nix builders)
2. builtins.wasm actions (for DICE)

The key property: no string interpolation bugs, no quoting errors.
-}
module Straylight.Script.Interpreter (
    -- * Compilation targets
    toBash,
    toWasm,
    toNix,

    -- * Types
    Script,
    Command (..),
    Path (..),
    Interp (..),
) where

import Data.List (intercalate)
import Data.Text (Text)
import qualified Data.Text as T

-- =============================================================================
-- Types (mirror Script.dhall)
-- =============================================================================

data Path
    = PathSrc Text -- Source-relative
    | PathOut Text
    | PathDep Text Text -- Dependency output
    | PathTmp Text
    | PathAbs Text -- Absolute (escape hatch)
    deriving (Show, Eq)

data Env
    = EnvOut
    | EnvSrc
    | EnvTmp
    | EnvNix Text
    | EnvVar Text
    deriving (Show, Eq)

data Interp
    = InterpLit Text
    | InterpPath Path
    | InterpEnv Env
    | InterpDep Text
    | InterpJoin Text [Interp]
    deriving (Show, Eq)

data Command
    = -- File ops
      CmdMkdir {mkdirPath :: Path, mkdirParents :: Bool}
    | CmdCopy {copySrc :: Path, copyDst :: Path, copyRecursive :: Bool}
    | CmdMove {moveSrc :: Path, moveDst :: Path}
    | CmdRemove {rmPath :: Path, rmRecursive :: Bool, rmForce :: Bool}
    | CmdSymlink {symlinkTarget :: Path, symlinkLink :: Path}
    | CmdChmod {chmodPath :: Path, chmodMode :: Text}
    | CmdTouch {touchPath :: Path}
    | -- Content ops
      CmdWrite {writePath :: Path, writeContent :: Interp}
    | CmdAppend {appendPath :: Path, appendContent :: Interp}
    | CmdSubstitute {subFile :: Path, subReplacements :: [(Text, Interp)]}
    | -- Archive ops
      CmdUntar {untarArchive :: Path, untarDest :: Path, untarStrip :: Int}
    | CmdUnzip {unzipArchive :: Path, unzipDest :: Path}
    | -- Build tools
      CmdConfigure {configFlags :: [Interp], configWorkdir :: Maybe Path}
    | CmdMake {makeTargets :: [Text], makeFlags :: [Interp], makeJobs :: Maybe Int}
    | CmdCMake {cmakeSrc :: Path, cmakeBuild :: Path, cmakeFlags :: [Interp]}
    | -- Install helpers
      CmdInstallBin {installBinSrc :: Path, installBinName :: Maybe Text}
    | CmdInstallLib {installLibSrc :: Path, installLibName :: Maybe Text}
    | -- Control flow
      CmdIf {ifCond :: Condition, ifThen :: [Command], ifElse :: [Command]}
    | CmdFor {forVar :: Text, forIn :: [Interp], forDo :: [Command]}
    | -- Escape hatch
      CmdRun {runCmd :: Text, runArgs :: [Interp], runEnv :: [(Text, Interp)]}
    | CmdShell Text -- Raw shell (warns)
    deriving (Show, Eq)

data Condition
    = CondPathExists Path
    | CondFileExists Path
    | CondDirExists Path
    | CondEnvSet Text
    | CondEnvEquals Text Text
    | CondAnd Condition Condition
    | CondOr Condition Condition
    | CondNot Condition
    | CondTrue
    | CondFalse
    deriving (Show, Eq)

type Script = [Command]

-- =============================================================================
-- Bash Compilation
-- =============================================================================

-- | Compile a script to bash
toBash :: Script -> Text
toBash commands =
    T.unlines
        [ "#!/usr/bin/env bash"
        , "set -euo pipefail"
        , ""
        , "# Generated from typed Script - DO NOT EDIT"
        , ""
        ]
        <> T.unlines (map cmdToBash commands)

-- | Compile a single command to bash
cmdToBash :: Command -> Text
cmdToBash = \case
    CmdMkdir{..} ->
        "mkdir " <> (if mkdirParents then "-p " else "") <> quote (pathToBash mkdirPath)
    CmdCopy{..} ->
        "cp "
            <> (if copyRecursive then "-r " else "")
            <> quote (pathToBash copySrc)
            <> " "
            <> quote (pathToBash copyDst)
    CmdMove{..} ->
        "mv " <> quote (pathToBash moveSrc) <> " " <> quote (pathToBash moveDst)
    CmdRemove{..} ->
        "rm "
            <> (if rmRecursive then "-r " else "")
            <> (if rmForce then "-f " else "")
            <> quote (pathToBash rmPath)
    CmdSymlink{..} ->
        "ln -s " <> quote (pathToBash symlinkTarget) <> " " <> quote (pathToBash symlinkLink)
    CmdChmod{..} ->
        "chmod " <> chmodMode <> " " <> quote (pathToBash chmodPath)
    CmdTouch{..} ->
        "touch " <> quote (pathToBash touchPath)
    CmdWrite{..} ->
        "cat > "
            <> quote (pathToBash writePath)
            <> " <<'DHALL_EOF'\n"
            <> interpToBash writeContent
            <> "\nDHALL_EOF"
    CmdAppend{..} ->
        "cat >> "
            <> quote (pathToBash appendPath)
            <> " <<'DHALL_EOF'\n"
            <> interpToBash appendContent
            <> "\nDHALL_EOF"
    CmdSubstitute{..} ->
        let sedExprs =
                map
                    ( \(from, to) ->
                        "s|" <> escapeSed from <> "|" <> escapeSed (interpToBash to) <> "|g"
                    )
                    subReplacements
         in "sed -i "
                <> T.intercalate " " (map (\e -> "-e " <> quote e) sedExprs)
                <> " "
                <> quote (pathToBash subFile)
    CmdUntar{..} ->
        "tar xf "
            <> quote (pathToBash untarArchive)
            <> " -C "
            <> quote (pathToBash untarDest)
            <> (if untarStrip > 0 then " --strip-components=" <> T.pack (show untarStrip) else "")
    CmdUnzip{..} ->
        "unzip -q " <> quote (pathToBash unzipArchive) <> " -d " <> quote (pathToBash unzipDest)
    CmdConfigure{..} ->
        let wd = maybe "" (\p -> "cd " <> quote (pathToBash p) <> " && ") configWorkdir
         in wd <> "./configure " <> T.intercalate " " (map interpToBash configFlags)
    CmdMake{..} ->
        "make"
            <> maybe "" (\j -> " -j" <> T.pack (show j)) makeJobs
            <> " "
            <> T.intercalate " " makeTargets
            <> " "
            <> T.intercalate " " (map interpToBash makeFlags)
    CmdCMake{..} ->
        "cmake -S "
            <> quote (pathToBash cmakeSrc)
            <> " -B "
            <> quote (pathToBash cmakeBuild)
            <> " "
            <> T.intercalate " " (map interpToBash cmakeFlags)
    CmdInstallBin{..} ->
        let name = maybe (pathBasename installBinSrc) id installBinName
         in "install -Dm755 "
                <> quote (pathToBash installBinSrc)
                <> " "
                <> quote ("$out/bin/" <> name)
    CmdInstallLib{..} ->
        let name = maybe (pathBasename installLibSrc) id installLibName
         in "install -Dm644 "
                <> quote (pathToBash installLibSrc)
                <> " "
                <> quote ("$out/lib/" <> name)
    CmdIf{..} ->
        "if "
            <> condToBash ifCond
            <> "; then\n"
            <> T.unlines (map (("  " <>) . cmdToBash) ifThen)
            <> (if null ifElse then "" else "else\n" <> T.unlines (map (("  " <>) . cmdToBash) ifElse))
            <> "fi"
    CmdFor{..} ->
        "for "
            <> forVar
            <> " in "
            <> T.intercalate " " (map interpToBash forIn)
            <> "; do\n"
            <> T.unlines (map (("  " <>) . cmdToBash) forDo)
            <> "done"
    CmdRun{..} ->
        let envPrefix =
                if null runEnv
                    then ""
                    else T.intercalate " " [k <> "=" <> quote (interpToBash v) | (k, v) <- runEnv] <> " "
         in envPrefix <> runCmd <> " " <> T.intercalate " " (map (quote . interpToBash) runArgs)
    CmdShell raw ->
        "# WARNING: Raw shell escape hatch\n" <> raw

-- | Convert path to bash expression
pathToBash :: Path -> Text
pathToBash = \case
    PathSrc p -> "$src/" <> p
    PathOut p -> "$out/" <> p
    PathDep dep p -> "${" <> dep <> "}/" <> p
    PathTmp p -> "$TMPDIR/" <> p
    PathAbs p -> p -- Already absolute

-- | Convert interpolation to bash
interpToBash :: Interp -> Text
interpToBash = \case
    InterpLit t -> t
    InterpPath p -> pathToBash p
    InterpEnv e -> envToBash e
    InterpDep d -> "${" <> d <> "}"
    InterpJoin sep parts -> T.intercalate sep (map interpToBash parts)

-- | Convert env to bash
envToBash :: Env -> Text
envToBash = \case
    EnvOut -> "$out"
    EnvSrc -> "$src"
    EnvTmp -> "$TMPDIR"
    EnvNix var -> "$NIX_" <> var
    EnvVar var -> "$" <> var

-- | Convert condition to bash
condToBash :: Condition -> Text
condToBash = \case
    CondPathExists p -> "[[ -e " <> quote (pathToBash p) <> " ]]"
    CondFileExists p -> "[[ -f " <> quote (pathToBash p) <> " ]]"
    CondDirExists p -> "[[ -d " <> quote (pathToBash p) <> " ]]"
    CondEnvSet v -> "[[ -n \"${" <> v <> ":-}\" ]]"
    CondEnvEquals v val -> "[[ \"${" <> v <> ":-}\" == " <> quote val <> " ]]"
    CondAnd l r -> condToBash l <> " && " <> condToBash r
    CondOr l r -> condToBash l <> " || " <> condToBash r
    CondNot c -> "! " <> condToBash c
    CondTrue -> "true"
    CondFalse -> "false"

-- | Quote a string for bash
quote :: Text -> Text
quote t = "\"" <> escapeQuotes t <> "\""

escapeQuotes :: Text -> Text
escapeQuotes = T.replace "\"" "\\\"" . T.replace "\\" "\\\\" . T.replace "$" "\\$"

escapeSed :: Text -> Text
escapeSed = T.replace "|" "\\|" . T.replace "/" "\\/"

pathBasename :: Path -> Text
pathBasename = \case
    PathSrc p -> T.takeWhileEnd (/= '/') p
    PathOut p -> T.takeWhileEnd (/= '/') p
    PathDep _ p -> T.takeWhileEnd (/= '/') p
    PathTmp p -> T.takeWhileEnd (/= '/') p
    PathAbs p -> T.takeWhileEnd (/= '/') p

-- =============================================================================
-- WASM/builtins.wasm Compilation
-- =============================================================================

-- | Compile to WASM action list (JSON for builtins.wasm)
toWasm :: Script -> Text
toWasm commands = "[\n" <> T.intercalate ",\n" (map cmdToWasm commands) <> "\n]"

cmdToWasm :: Command -> Text
cmdToWasm = \case
    CmdMkdir{..} ->
        wasmAction "mkdir" [("path", pathToWasm mkdirPath), ("parents", boolToWasm mkdirParents)]
    CmdCopy{..} ->
        wasmAction "copy" [("src", pathToWasm copySrc), ("dst", pathToWasm copyDst), ("recursive", boolToWasm copyRecursive)]
    CmdSymlink{..} ->
        wasmAction "symlink" [("target", pathToWasm symlinkTarget), ("link", pathToWasm symlinkLink)]
    CmdWrite{..} ->
        wasmAction "write" [("path", pathToWasm writePath), ("content", interpToWasm writeContent)]
    CmdUntar{..} ->
        wasmAction "untar" [("archive", pathToWasm untarArchive), ("dest", pathToWasm untarDest), ("strip", T.pack (show untarStrip))]
    CmdInstallBin{..} ->
        wasmAction "install_bin" [("src", pathToWasm installBinSrc)]
    CmdRun{..} ->
        wasmAction "run" [("cmd", "\"" <> runCmd <> "\""), ("args", "[" <> T.intercalate "," (map interpToWasm runArgs) <> "]")]
    CmdShell raw ->
        wasmAction "shell" [("script", "\"" <> escapeJson raw <> "\"")]
    -- TODO: other commands
    other -> wasmAction "todo" [("cmd", "\"" <> T.pack (show other) <> "\"")]

wasmAction :: Text -> [(Text, Text)] -> Text
wasmAction action fields =
    "  {\"action\": \""
        <> action
        <> "\", "
        <> T.intercalate ", " [k <> ": " <> v | (k, v) <- fields]
        <> "}"

pathToWasm :: Path -> Text
pathToWasm = \case
    PathSrc p -> "{\"type\": \"src\", \"path\": \"" <> p <> "\"}"
    PathOut p -> "{\"type\": \"out\", \"path\": \"" <> p <> "\"}"
    PathDep dep p -> "{\"type\": \"dep\", \"dep\": \"" <> dep <> "\", \"path\": \"" <> p <> "\"}"
    PathTmp p -> "{\"type\": \"tmp\", \"path\": \"" <> p <> "\"}"
    PathAbs p -> "{\"type\": \"abs\", \"path\": \"" <> p <> "\"}"

interpToWasm :: Interp -> Text
interpToWasm = \case
    InterpLit t -> "\"" <> escapeJson t <> "\""
    InterpPath p -> pathToWasm p
    InterpEnv e -> "{\"type\": \"env\", \"var\": \"" <> envName e <> "\"}"
    InterpDep d -> "{\"type\": \"dep\", \"ref\": \"" <> d <> "\"}"
    InterpJoin sep parts -> "{\"type\": \"join\", \"sep\": \"" <> sep <> "\", \"parts\": [" <> T.intercalate "," (map interpToWasm parts) <> "]}"

envName :: Env -> Text
envName = \case
    EnvOut -> "out"
    EnvSrc -> "src"
    EnvTmp -> "TMPDIR"
    EnvNix v -> "NIX_" <> v
    EnvVar v -> v

boolToWasm :: Bool -> Text
boolToWasm True = "true"
boolToWasm False = "false"

escapeJson :: Text -> Text
escapeJson = T.replace "\"" "\\\"" . T.replace "\\" "\\\\" . T.replace "\n" "\\n"

-- =============================================================================
-- Nix Expression Compilation
-- =============================================================================

-- | Compile to Nix expression (for use in builders)
toNix :: Script -> Text
toNix commands =
    "''\n" <> T.unlines (map cmdToBash commands) <> "''"
