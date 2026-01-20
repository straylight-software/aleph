#!/usr/bin/env runghc
{-# LANGUAGE OverloadedStrings #-}

{- |
Unified wrapper generator - auto-detects clap vs GNU getopt format

Usage:
  ./gen-wrapper.hs <command>              # prints to stdout
  ./gen-wrapper.hs <command> --write      # writes to Weyl/Script/Tools/<Name>.hs
  ./gen-wrapper.hs <command> <ModuleName> # custom module name

Examples:
  ./gen-wrapper.hs rg                     # auto-detect, print to stdout
  ./gen-wrapper.hs rg --write             # write to Weyl/Script/Tools/Rg.hs
  ./gen-wrapper.hs grep --write           # GNU tool, auto-detected
  ./gen-wrapper.hs jq Jq --write          # custom name, write to file

Detection heuristics:
  - Clap: Options section with "-x, --long <ARG>" format
  - GNU:  Options with "  -x, --long=ARG" or "      --long" format
-}
module Main where

import Control.Monad (when)
import Data.Char (isSpace, toUpper)
import Data.List (isInfixOf)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.Directory (doesFileExist)
import System.Environment (getArgs, getProgName)
import System.Exit (ExitCode (..), exitFailure)
import System.IO (hPutStrLn, stderr)
import System.Process (readProcess, readProcessWithExitCode)

import qualified Aleph.Script.Clap as Clap
import qualified Aleph.Script.Getopt as Getopt

data HelpFormat = ClapFormat | GnuFormat
    deriving (Show, Eq)

data Config = Config
    { cfgCommand :: String
    , cfgModuleName :: Text
    , cfgWriteFile :: Bool
    , cfgForceFormat :: Maybe HelpFormat
    }
    deriving (Show)

main :: IO ()
main = do
    args <- getArgs
    case parseArgs args of
        Nothing -> usage
        Just cfg -> runGenerator cfg

parseArgs :: [String] -> Maybe Config
parseArgs args = case filter (not . isFlag) args of
    [] -> Nothing
    [cmd] ->
        Just
            Config
                { cfgCommand = cmd
                , cfgModuleName = defaultModuleName cmd
                , cfgWriteFile = hasFlag "--write" args
                , cfgForceFormat = parseForceFormat args
                }
    [cmd, modName] ->
        Just
            Config
                { cfgCommand = cmd
                , cfgModuleName = T.pack modName
                , cfgWriteFile = hasFlag "--write" args
                , cfgForceFormat = parseForceFormat args
                }
    _ -> Nothing
  where
    isFlag s = s `elem` ["--write", "--clap", "--gnu"]
    hasFlag f = elem f
    parseForceFormat as
        | "--clap" `elem` as = Just ClapFormat
        | "--gnu" `elem` as = Just GnuFormat
        | otherwise = Nothing

usage :: IO ()
usage = do
    prog <- getProgName
    hPutStrLn stderr $
        unlines
            [ "Usage: " ++ prog ++ " <command> [module-name] [options]"
            , ""
            , "Generate typed Haskell wrapper for CLI tools."
            , "Auto-detects clap (Rust) vs GNU getopt format."
            , ""
            , "Options:"
            , "  --write    Write to Weyl/Script/Tools/<Name>.hs instead of stdout"
            , "  --clap     Force clap format parsing"
            , "  --gnu      Force GNU getopt format parsing"
            , ""
            , "Examples:"
            , "  " ++ prog ++ " rg                    # auto-detect, stdout"
            , "  " ++ prog ++ " rg --write            # write to Rg.hs"
            , "  " ++ prog ++ " find --gnu --write    # force GNU format"
            , "  " ++ prog ++ " jq Jq --clap --write  # custom name, force clap"
            ]
    exitFailure

defaultModuleName :: String -> Text
defaultModuleName [] = "Unknown"
defaultModuleName (c : cs) = T.pack (toUpper c : cs)

runGenerator :: Config -> IO ()
runGenerator cfg = do
    -- Try --help first, fall back to -h
    (exitCode, helpText, _) <- readProcessWithExitCode (cfgCommand cfg) ["--help"] ""

    finalHelp <- case exitCode of
        ExitSuccess -> return helpText
        _ -> do
            -- Some tools only support -h
            (exit2, help2, _) <- readProcessWithExitCode (cfgCommand cfg) ["-h"] ""
            case exit2 of
                ExitSuccess -> return help2
                _ -> do
                    hPutStrLn stderr $ "Error: Could not get help from " ++ cfgCommand cfg
                    hPutStrLn stderr $ "Tried: " ++ cfgCommand cfg ++ " --help"
                    hPutStrLn stderr $ "Tried: " ++ cfgCommand cfg ++ " -h"
                    exitFailure

    let format = case cfgForceFormat cfg of
            Just f -> f
            Nothing -> detectFormat finalHelp
        helpT = T.pack finalHelp
        modName = cfgModuleName cfg
        cmdName = T.pack (cfgCommand cfg)

    hPutStrLn stderr $ "Detected format: " ++ show format

    let generated = case format of
            ClapFormat ->
                let parsed = Clap.parseHelp helpT
                 in Clap.generateModule modName cmdName parsed
            GnuFormat ->
                let parsed = Getopt.parseHelp helpT
                 in Getopt.generateModule modName cmdName parsed

    if cfgWriteFile cfg
        then do
            let outPath = "Weyl/Script/Tools/" ++ T.unpack modName ++ ".hs"
            exists <- doesFileExist outPath
            when exists $ do
                hPutStrLn stderr $ "Warning: Overwriting existing file: " ++ outPath
            TIO.writeFile outPath generated
            hPutStrLn stderr $ "Wrote: " ++ outPath
        else TIO.putStr generated

{- | Detect whether help text is clap or GNU getopt format

Key differences:
  Clap: "-e PATTERN, --regexp=PATTERN" (space before value, comma separator)
        "POSITIONAL ARGUMENTS:" or "Arguments:" sections
        Often has "USAGE:" in caps

  GNU:  "  -e, --regexp=PATTERN" (no space, value after =)
        "      --long-only" (long-only options with 6+ space indent)
        Description on same line or continuation
-}
detectFormat :: String -> HelpFormat
detectFormat help
    -- Strong clap indicators
    | hasPositionalArgs = ClapFormat
    | hasUsageSection && hasAngleBracketArgs = ClapFormat
    -- Clap-style: "-x ARG, --long=ARG" (space before ARG in short form)
    | hasClapShortArgStyle = ClapFormat
    -- GNU-style: long-only options with deep indent
    | hasLongOnlyOptions = GnuFormat
    -- GNU coreutils style
    | hasGnuStyleOptions = GnuFormat
    -- Default based on common patterns
    | hasAngleBracketArgs = ClapFormat
    | otherwise = GnuFormat
  where
    helpLines = lines help

    -- Clap uses "POSITIONAL ARGUMENTS:", "Arguments:", or "Usage:"
    hasPositionalArgs =
        "POSITIONAL ARGUMENTS:" `isInfixOf` help
            || "\nArguments:" `isInfixOf` help
            || "\nArguments\n" `isInfixOf` help
    hasUsageSection = "USAGE:" `isInfixOf` help || "Usage:" `isInfixOf` help

    -- Clap uses <ARG> style in option lines
    hasAngleBracketArgs = any isAngleBracketLine helpLines
    isAngleBracketLine line =
        "  -" `isInfixOf` line && "<" `isInfixOf` line && ">" `isInfixOf` line

    -- Clap short option style: "-x ARG" with space before ARG
    -- e.g., "-e PATTERN, --regexp=PATTERN"
    hasClapShortArgStyle = any isClapShortArg helpLines
    isClapShortArg line =
        let trimmed = dropWhile isSpace line
         in take 1 trimmed == "-"
                &&
                -- Has pattern like "-x ARG," where ARG is uppercase
                any (\w -> all (`elem` ['A' .. 'Z']) w && length w > 1) (words (take 20 trimmed))

    -- GNU has long-only options (6+ spaces before --)
    hasLongOnlyOptions = any isLongOnlyLine helpLines
    isLongOnlyLine line =
        let indent = length (takeWhile isSpace line)
            stripped = dropWhile isSpace line
         in indent >= 6 && take 2 stripped == "--"

    -- GNU style: "  -x, --long" without angle brackets
    hasGnuStyleOptions = any isGnuStyleLine helpLines
    isGnuStyleLine line =
        let trimmed = dropWhile isSpace line
         in take 1 trimmed == "-"
                && ", --" `isInfixOf` take 20 line
                && not ("<" `isInfixOf` take 30 line)
