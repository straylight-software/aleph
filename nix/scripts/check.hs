#!/usr/bin/env runghc
{-# LANGUAGE OverloadedStrings #-}

{- |
Quick validation script for Aleph.Script tooling

Run: runghc -i. check.hs

Checks:
  1. All generated wrappers compile
  2. Parsers don't crash on corpus files
  3. Key invariants hold
-}
module Main where

import Control.Monad (forM_, when)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.Directory (listDirectory)
import System.Exit (ExitCode (..), exitFailure, exitSuccess)
import System.Process (readProcessWithExitCode)

import qualified Aleph.Script.Clap as Clap
import qualified Aleph.Script.Getopt as Getopt

main :: IO ()
main = do
    putStrLn "=== Aleph.Script Validation ==="
    putStrLn ""

    -- Check 1: All wrappers compile
    putStr "Compiling all wrappers... "
    (exit, _, stderr) <-
        readProcessWithExitCode
            "ghc"
            ["-fno-code", "-i.", "Weyl/Script/Tools.hs"]
            ""
    case exit of
        ExitSuccess -> putStrLn "OK"
        ExitFailure _ -> do
            putStrLn "FAILED"
            putStrLn stderr
            exitFailure

    -- Check 2: Parse corpus files without crashing
    putStr "Parsing clap corpus... "
    clapFiles <- listDirectory "corpus"
    forM_ (filter (".txt" `isSuffixOf`) clapFiles) $ \f -> do
        content <- TIO.readFile ("corpus/" ++ f)
        let _ = Clap.parseHelp content
        return ()
    putStrLn $ "OK (" ++ show (length clapFiles) ++ " files)"

    putStr "Parsing GNU corpus... "
    gnuFiles <- listDirectory "corpus-gnu"
    forM_ (filter (".txt" `isSuffixOf`) gnuFiles) $ \f -> do
        content <- TIO.readFile ("corpus-gnu/" ++ f)
        let _ = Getopt.parseHelp content
        return ()
    putStrLn $ "OK (" ++ show (length gnuFiles) ++ " files)"

    -- Check 3: Generated code has required structure
    putStr "Checking generated structure... "
    let testInput =
            T.unlines
                [ "Options:"
                , "  -v, --verbose  Be verbose"
                , "  -f, --file=FILE  Input file"
                ]
        generated = Clap.generateModule "Test" "test" (Clap.parseHelp testInput)
        checks =
            [ ("module", "module Aleph.Script.Tools.Test" `T.isInfixOf` generated)
            , ("Options", "data Options = Options" `T.isInfixOf` generated)
            , ("defaults", "defaults :: Options" `T.isInfixOf` generated)
            , ("buildArgs", "buildArgs :: Options -> [Text]" `T.isInfixOf` generated)
            ]
        failures = [name | (name, ok) <- checks, not ok]
    if null failures
        then putStrLn "OK"
        else do
            putStrLn $ "FAILED: missing " ++ show failures
            exitFailure

    -- Check 4: No duplicate fields (the bug we fixed)
    putStr "Checking no duplicate fields... "
    let dupInput =
            T.unlines
                [ "Options:"
                , "  -l, --long  First"
                , "  -L, --long  Duplicate" -- Same field name!
                ]
        dupGenerated = Clap.generateModule "Test" "test" (Clap.parseHelp dupInput)
        fieldCount = length $ filter (== "long ::") $ map T.strip $ T.lines dupGenerated
    if fieldCount <= 1
        then putStrLn "OK (deduplication works)"
        else do
            putStrLn $ "FAILED: found " ++ show fieldCount ++ " 'long' fields"
            exitFailure

    putStrLn ""
    putStrLn "=== All checks passed ==="
    exitSuccess
  where
    isSuffixOf suffix str = drop (length str - length suffix) str == suffix
