{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module      : Main (nix-toolchain)
Description : Extract toolchain from Nix stdenv

Usage:
  nix-toolchain env nixpkgs#stdenv     -> CC=... CXX=... format
  nix-toolchain flags nixpkgs#stdenv   -> compiler flags only
  nix-toolchain json nixpkgs#stdenv    -> full JSON toolchain

Buck2 calls this to get toolchain info without reimplementing
stdenv logic in Starlark.
-}
module Main where

import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Lazy.Char8 as BL8
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.IO (hPutStrLn, stderr)

import Armitage.Toolchain

data Mode
  = ModeEnv Text    -- ^ Output as environment variables
  | ModeFlags Text  -- ^ Output as compiler flags
  | ModeJson Text   -- ^ Output as JSON
  deriving (Show, Eq)

main :: IO ()
main = do
  args <- getArgs
  mode <- parseArgs args
  result <- runMode mode
  case result of
    Left err -> do
      TIO.hPutStrLn stderr err
      exitFailure
    Right output -> TIO.putStr output

parseArgs :: [String] -> IO Mode
parseArgs = \case
  ["env", ref] -> pure $ ModeEnv (T.pack ref)
  ["flags", ref] -> pure $ ModeFlags (T.pack ref)
  ["json", ref] -> pure $ ModeJson (T.pack ref)
  _ -> do
    hPutStrLn stderr "Usage: nix-toolchain <mode> <flake-ref>"
    hPutStrLn stderr ""
    hPutStrLn stderr "Modes:"
    hPutStrLn stderr "  env <ref>    Output as CC=... CXX=... format"
    hPutStrLn stderr "  flags <ref>  Output as compiler flags"
    hPutStrLn stderr "  json <ref>   Output as JSON"
    hPutStrLn stderr ""
    hPutStrLn stderr "Examples:"
    hPutStrLn stderr "  nix-toolchain env nixpkgs#stdenv"
    hPutStrLn stderr "  nix-toolchain flags nixpkgs#llvmPackages_18.stdenv"
    hPutStrLn stderr "  nix-toolchain json nixpkgs#hello  # uses hello.stdenv"
    exitFailure

runMode :: Mode -> IO (Either Text Text)
runMode = \case
  ModeEnv ref -> do
    result <- extractToolchain ref
    pure $ toolchainToEnv <$> result
  ModeFlags ref -> do
    result <- extractToolchain ref
    pure $ toolchainToFlags <$> result
  ModeJson ref -> do
    result <- extractToolchain ref
    pure $ (T.pack . BL8.unpack . Aeson.encode) <$> result
