#!/usr/bin/env runghc
{-# LANGUAGE OverloadedStrings #-}

{- |
Generate typed Haskell wrapper for a GNU getopt_long CLI tool

Usage:
  ./gen-gnu-wrapper.hs <command> [module-name]

Examples:
  ./gen-gnu-wrapper.hs ls              # generates Ls.hs
  ./gen-gnu-wrapper.hs grep Grep       # generates Grep.hs
  ./gen-gnu-wrapper.hs tar > Tar.hs
-}
module Main where

import Data.Char (toUpper)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.Environment (getArgs, getProgName)
import System.Exit (exitFailure)
import System.IO (hPutStrLn, stderr)
import System.Process (readProcess)

import Aleph.Script.Getopt (generateModule, parseHelp)

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
            , "Generate typed Haskell wrapper for a GNU getopt_long CLI tool."
            , ""
            , "Examples:"
            , "  " ++ prog ++ " ls              # generates wrapper for ls"
            , "  " ++ prog ++ " grep Grep       # use 'Grep' as module name"
            , "  " ++ prog ++ " tar > Tar.hs    # save to file"
            ]
    exitFailure

-- | Generate default module name from command
defaultModuleName :: String -> Text
defaultModuleName cmd = case cmd of
    [] -> "Unknown"
    (c : cs) -> T.pack (toUpper c : cs)

-- | Run the generator
generateWrapper :: String -> Text -> IO ()
generateWrapper cmd moduleName = do
    helpText <- readProcess cmd ["--help"] ""
    let parsed = parseHelp (T.pack helpText)
        generated = generateModule moduleName (T.pack cmd) parsed
    TIO.putStr generated
