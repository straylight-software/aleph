#!/usr/bin/env runghc
{-# LANGUAGE OverloadedStrings #-}

-- \|
-- Generate typed Haskell wrapper for a clap-based CLI tool
--
-- Usage:
--   ./gen-tool-wrapper.hs <command> [module-name]
--
-- Examples:
--   ./gen-tool-wrapper.hs rg              # generates Rg.hs
--   ./gen-tool-wrapper.hs rg Ripgrep      # generates Ripgrep.hs
--   ./gen-tool-wrapper.hs fd > Fd.hs
--
-- The script runs "<command> --help" to get the help text, parses it,
-- and generates a Haskell module to stdout.
--
-- Requirements:
--   nix-shell -p "haskellPackages.ghcWithPackages (p: [p.megaparsec p.text])"

import Data.Char (toUpper)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.Environment (getArgs, getProgName)
import System.Exit (exitFailure)
import System.IO (hPutStrLn, stderr)
import System.Process (readProcess)

import Aleph.Script.Clap (generateModule, parseHelp)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [] -> usage
        [cmd] -> generateWrapper cmd (defaultModuleName cmd)
        [cmd, modName] -> generateWrapper cmd (T.pack modName)
        _ -> usage

usage :: IO ()
usage = do
    prog <- getProgName
    hPutStrLn stderr $
        unlines
            [ "Usage: " ++ prog ++ " <command> [module-name]"
            , ""
            , "Generate typed Haskell wrapper for a clap-based CLI tool."
            , ""
            , "Examples:"
            , "  " ++ prog ++ " rg              # generates wrapper for ripgrep"
            , "  " ++ prog ++ " rg Ripgrep      # use 'Ripgrep' as module name"
            , "  " ++ prog ++ " fd > Fd.hs      # save to file"
            , ""
            , "The script runs '<command> --help' and generates Haskell code to stdout."
            ]
    exitFailure

{- | Generate default module name from command
e.g., "rg" -> "Rg", "delta" -> "Delta"
-}
defaultModuleName :: String -> Text
defaultModuleName cmd = case cmd of
    [] -> "Unknown"
    (c : cs) -> T.pack (toUpper c : cs)

-- | Run the generator
generateWrapper :: String -> Text -> IO ()
generateWrapper cmd moduleName = do
    -- Get help text from the command
    helpText <- readProcess cmd ["--help"] ""

    -- Parse and generate
    let parsed = parseHelp (T.pack helpText)
        generated = generateModule moduleName (T.pack cmd) parsed

    TIO.putStr generated
