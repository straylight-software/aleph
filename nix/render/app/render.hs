{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- |
-- render - CLI for render.nix type inference
--
-- Usage:
--   render parse <script>       Parse and show facts
--   render infer <script>       Infer types and show schema
--   render check <script>       Check for policy violations
--   render emit <script> [fmt]  Generate emit-config function
module Main where

import Data.Aeson (encode)
import qualified Data.ByteString.Lazy.Char8 as BL
import qualified Data.Text.IO as TIO
import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)
import Render
import Render.Emit.Config (emitConfigFunction)

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["parse", file] -> cmdParse file
    ["infer", file] -> cmdInfer file
    ["check", file] -> cmdCheck file
    ["emit", file] -> cmdEmit file
    ["--help"] -> usage
    ["-h"] -> usage
    [] -> usage
    _ -> do
      putStrLn $ "Unknown command: " ++ unwords args
      usage
      exitFailure

usage :: IO ()
usage = do
  putStrLn "render - bash type inference for render.nix"
  putStrLn ""
  putStrLn "Usage:"
  putStrLn "  render parse <script.sh>   Parse and show extracted facts"
  putStrLn "  render infer <script.sh>   Infer types and show schema (JSON)"
  putStrLn "  render check <script.sh>   Check for policy violations"
  putStrLn "  render emit <script.sh>    Generate emit-config bash function"
  putStrLn ""
  putStrLn "Examples:"
  putStrLn "  render infer ./deploy.sh | jq '.env'"
  putStrLn "  render check ./scripts/*.sh"
  putStrLn "  render emit ./deploy.sh >> deploy.sh"

cmdParse :: FilePath -> IO ()
cmdParse file = do
  result <- parseScriptFile file
  case result of
    Left err -> do
      TIO.putStrLn $ "Parse error: " <> err
      exitFailure
    Right script -> do
      putStrLn "Facts:"
      mapM_ print (scriptFacts script)

cmdInfer :: FilePath -> IO ()
cmdInfer file = do
  result <- parseScriptFile file
  case result of
    Left err -> do
      TIO.putStrLn $ "Error: " <> err
      exitFailure
    Right script -> do
      BL.putStrLn (encode (scriptSchema script))

cmdCheck :: FilePath -> IO ()
cmdCheck file = do
  result <- parseScriptFile file
  case result of
    Left err -> do
      TIO.putStrLn $ "Error: " <> err
      exitFailure
    Right script -> do
      let schema = scriptSchema script
      let bareCount = length (schemaBareCommands schema)
      let dynCount = length (schemaDynamicCommands schema)
      if bareCount > 0 || dynCount > 0
        then do
          putStrLn $ "Policy violations in " ++ file ++ ":"
          mapM_ (\cmd -> putStrLn $ "  bare command: " ++ show cmd) (schemaBareCommands schema)
          mapM_ (\cmd -> putStrLn $ "  dynamic command: $" ++ show cmd) (schemaDynamicCommands schema)
          exitFailure
        else do
          putStrLn $ file ++ ": OK"
          exitSuccess

cmdEmit :: FilePath -> IO ()
cmdEmit file = do
  result <- parseScriptFile file
  case result of
    Left err -> do
      TIO.putStrLn $ "Error: " <> err
      exitFailure
    Right script -> do
      TIO.putStr $ emitConfigFunction (scriptSchema script)
