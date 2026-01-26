{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module      : Main (nix-analyze)
Description : CLI for nix flake analysis

Modes:

  nix-analyze resolve nixpkgs#zlib nixpkgs#openssl ...
    -> Compiler flags, one per line (preferred - no JSON)
    
  nix-analyze unroll nixpkgs#zlib
    -> JSON with build info (TODO: replace JSON)
    
  nix-analyze deps nixpkgs#zlib
    -> JSON with dependencies (TODO: replace JSON)

Buck2 calls this tool to get nix package information.
-}
module Main where

import qualified Data.ByteString.Lazy.Char8 as BL8
import Data.Text (Text)
import qualified Data.Text as T
import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.IO (hPutStrLn, stderr)

import Armitage.Nix (AnalysisMode (..), runAnalysis)

main :: IO ()
main = do
  args <- getArgs
  mode <- parseArgs args
  result <- runAnalysis mode
  BL8.putStr result

parseArgs :: [String] -> IO AnalysisMode
parseArgs = \case
  ("resolve" : refs) | not (null refs) -> 
    pure $ ModeResolve (map T.pack refs)
  ["unroll", ref] -> 
    pure $ ModeUnroll (T.pack ref)
  ["deps", ref] -> 
    pure $ ModeDeps (T.pack ref)
  _ -> do
    hPutStrLn stderr "Usage: nix-analyze <mode> <args...>"
    hPutStrLn stderr ""
    hPutStrLn stderr "Modes:"
    hPutStrLn stderr "  resolve <refs...>  Output compiler flags (preferred)"
    hPutStrLn stderr "  unroll <ref>       Unroll build system (JSON)"
    hPutStrLn stderr "  deps <ref>         Get dependencies (JSON)"
    hPutStrLn stderr ""
    hPutStrLn stderr "Examples:"
    hPutStrLn stderr "  nix-analyze resolve nixpkgs#zlib nixpkgs#openssl"
    hPutStrLn stderr "  nix-analyze unroll nixpkgs#zlib"
    hPutStrLn stderr "  nix-analyze deps nixpkgs#openssl"
    exitFailure
