{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module      : Main
Description : Armitage CLI - daemon-free Nix operations

The armitage command-line tool provides daemon-free Nix operations:
  - armitage build <drv>    Build a derivation without daemon
  - armitage build-dhall    Build from Dhall target file
  - armitage proxy          Run the witness proxy
  - armitage store <cmd>    Store operations
  - armitage cas <cmd>      CAS operations

The daemon is hostile infrastructure. armitage routes around it.

Build:
  buck2 build //src/armitage:armitage

Usage:
  armitage build /nix/store/xxx.drv
  armitage build-dhall BUILD.dhall
  armitage proxy --port 8080
  armitage store add ./path
  armitage cas upload <hash> <file>
-}
module Main where

import Control.Exception (try, SomeException)
import Control.Monad (when, forM_)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.Environment (getArgs, getProgName)
import System.Exit (exitFailure, exitSuccess)
import System.IO (hPutStrLn, stderr)

import qualified Armitage.Dhall as Dhall
import qualified Armitage.DICE as DICE
import qualified Armitage.Trace as Trace

-- -----------------------------------------------------------------------------
-- Main
-- -----------------------------------------------------------------------------

main :: IO ()
main = do
  args <- getArgs
  case args of
    [] -> usage
    ("build" : rest) -> cmdBuild rest
    ("build-dhall" : rest) -> cmdBuildDhall rest
    ("analyze" : rest) -> cmdAnalyze rest
    ("run" : rest) -> cmdRun rest
    ("trace" : rest) -> cmdTrace rest
    ("proxy" : rest) -> cmdProxy rest
    ("store" : rest) -> cmdStore rest
    ("cas" : rest) -> cmdCAS rest
    ("--help" : _) -> usage
    ("-h" : _) -> usage
    (cmd : _) -> do
      hPutStrLn stderr $ "Unknown command: " <> cmd
      usage
      exitFailure

-- -----------------------------------------------------------------------------
-- Commands
-- -----------------------------------------------------------------------------

-- | Build command (from .drv file)
cmdBuild :: [String] -> IO ()
cmdBuild args = case args of
  [] -> do
    hPutStrLn stderr "Usage: armitage build <derivation.drv>"
    exitFailure
  (drvPath : _) -> do
    putStrLn $ "Building: " <> drvPath
    putStrLn "TODO: Implement daemon-free build"
    -- result <- Builder.runBuild defaultConfig drvPath
    -- case result of
    --   Left err -> do
    --     hPutStrLn stderr $ "Build failed: " <> show err
    --     exitFailure
    --   Right result -> do
    --     putStrLn $ "Build succeeded"
    --     forM_ (Map.toList $ Builder.brOutputs result) $ \(name, path) ->
    --       putStrLn $ "  " <> T.unpack name <> ": " <> show path

-- | Build command (from Dhall target file)
cmdBuildDhall :: [String] -> IO ()
cmdBuildDhall args = case args of
  [] -> do
    hPutStrLn stderr "Usage: armitage build-dhall <BUILD.dhall>"
    exitFailure
  (dhallPath : _) -> do
    putStrLn $ "Loading target from: " <> dhallPath
    result <- try $ Dhall.loadTarget dhallPath
    case result of
      Left (e :: SomeException) -> do
        hPutStrLn stderr $ "Failed to load Dhall: " <> show e
        exitFailure
      Right target -> do
        let tc = Dhall.toolchain target
        putStrLn $ "Target: " <> T.unpack (Dhall.targetName target)
        putStrLn $ "Triple: " <> T.unpack (Dhall.renderTriple (Dhall.target tc))
        case Dhall.renderGpu (Dhall.gpu (Dhall.target tc)) of
          Just sm -> putStrLn $ "GPU: " <> T.unpack sm
          Nothing -> pure ()
        putStrLn $ "Coeffects: " <> show (length (Dhall.requires target)) <> " resource(s)"
        forM_ (Dhall.requires target) $ \r ->
          putStrLn $ "  - " <> showResource r
        putStrLn ""
        putStrLn "Converting to derivation..."
        let drv = Dhall.targetToDerivation target
        putStrLn "TODO: Execute build"
  where
    showResource = \case
      Dhall.Resource_Pure -> "pure"
      Dhall.Resource_Network -> "network"
      Dhall.Resource_Auth p -> "auth:" <> T.unpack p
      Dhall.Resource_Sandbox s -> "sandbox:" <> T.unpack s
      Dhall.Resource_Filesystem p -> "fs:" <> T.unpack p

-- | Proxy command
cmdProxy :: [String] -> IO ()
cmdProxy args = do
  putStrLn "Starting witness proxy..."
  putStrLn "TODO: Import and run proxy Main"
  -- For now, just explain what would happen
  let port = parsePort args
  putStrLn $ "Would listen on port " <> show port
  putStrLn "All fetches would be:"
  putStrLn "  - Intercepted (TLS MITM)"
  putStrLn "  - Content-hashed"
  putStrLn "  - Cached in CAS"
  putStrLn "  - Logged as attestations"

-- | Store command
cmdStore :: [String] -> IO ()
cmdStore args = case args of
  [] -> do
    hPutStrLn stderr "Usage: armitage store <command>"
    hPutStrLn stderr "Commands:"
    hPutStrLn stderr "  add <path>       Add path to store"
    hPutStrLn stderr "  info <path>      Query path info"
    hPutStrLn stderr "  verify <path>    Verify path integrity"
    exitFailure
  ("add" : path : _) -> do
    putStrLn $ "Adding to store: " <> path
    putStrLn "TODO: Implement store add"
  ("info" : path : _) -> do
    putStrLn $ "Querying: " <> path
    putStrLn "TODO: Implement store info"
  ("verify" : path : _) -> do
    putStrLn $ "Verifying: " <> path
    putStrLn "TODO: Implement store verify"
  (cmd : _) -> do
    hPutStrLn stderr $ "Unknown store command: " <> cmd
    exitFailure

-- | CAS command
cmdCAS :: [String] -> IO ()
cmdCAS args = case args of
  [] -> do
    hPutStrLn stderr "Usage: armitage cas <command>"
    hPutStrLn stderr "Commands:"
    hPutStrLn stderr "  upload <file>    Upload blob to CAS"
    hPutStrLn stderr "  download <hash>  Download blob from CAS"
    hPutStrLn stderr "  exists <hash>    Check if blob exists"
    exitFailure
  ("upload" : path : _) -> do
    putStrLn $ "Uploading to CAS: " <> path
    putStrLn "TODO: Implement CAS upload"
  ("download" : hash : _) -> do
    putStrLn $ "Downloading from CAS: " <> hash
    putStrLn "TODO: Implement CAS download"
  ("exists" : hash : _) -> do
    putStrLn $ "Checking CAS: " <> hash
    putStrLn "TODO: Implement CAS exists"
  (cmd : _) -> do
    hPutStrLn stderr $ "Unknown CAS command: " <> cmd
    exitFailure

-- -----------------------------------------------------------------------------
-- Helpers
-- -----------------------------------------------------------------------------

parsePort :: [String] -> Int
parsePort = go 8080
 where
  go def [] = def
  go def ("--port" : p : _) = read p
  go def ("-p" : p : _) = read p
  go def (_ : rest) = go def rest

-- | Analyze command - resolve deps and build action graph
cmdAnalyze :: [String] -> IO ()
cmdAnalyze args = case args of
  [] -> do
    hPutStrLn stderr "Usage: armitage analyze <BUILD.dhall>"
    exitFailure
  (dhallPath : _) -> do
    putStrLn $ "Analyzing: " <> dhallPath
    target <- Dhall.loadTarget dhallPath
    
    putStrLn $ "Target: " <> T.unpack (Dhall.targetName target)
    putStrLn ""
    
    -- Show deps before resolution
    putStrLn "Dependencies:"
    forM_ (Dhall.deps target) $ \dep -> case dep of
      Dhall.Dep_Local t -> putStrLn $ "  local: " <> T.unpack t
      Dhall.Dep_Flake t -> putStrLn $ "  flake: " <> T.unpack t
      Dhall.Dep_PkgConfig t -> putStrLn $ "  pkg-config: " <> T.unpack t
      Dhall.Dep_External _ dn -> putStrLn $ "  external: " <> T.unpack dn
    putStrLn ""
    
    -- Analyze (resolves flakes)
    putStrLn "Resolving flake references..."
    result <- DICE.analyze target
    
    -- Show resolution results
    if null (DICE.arErrors result)
      then do
        putStrLn "Resolved:"
        forM_ (DICE.arFlakes result) $ \rf -> do
          putStrLn $ "  " <> T.unpack (DICE.rfRef rf) <> ":"
          forM_ (Map.toList $ DICE.rfOutputs rf) $ \(name, path) ->
            putStrLn $ "    " <> T.unpack name <> " -> " <> T.unpack path
        putStrLn ""
        
        -- Show action graph
        let graph = DICE.arGraph result
        putStrLn $ "Action graph: " <> show (length $ DICE.agActions graph) <> " action(s)"
        forM_ (DICE.topoSort graph) $ \key -> do
          let action = DICE.agActions graph Map.! key
          putStrLn $ "  " <> T.unpack (DICE.aIdentifier action) 
                   <> " [" <> show (DICE.aCategory action) <> "]"
      else do
        hPutStrLn stderr "Resolution errors:"
        forM_ (DICE.arErrors result) $ \e ->
          hPutStrLn stderr $ "  " <> T.unpack e
        exitFailure

-- | Run command - analyze and execute
cmdRun :: [String] -> IO ()
cmdRun args = case args of
  [] -> do
    hPutStrLn stderr "Usage: armitage run <BUILD.dhall>"
    exitFailure
  (dhallPath : _) -> do
    putStrLn $ "Loading: " <> dhallPath
    target <- Dhall.loadTarget dhallPath
    
    putStrLn $ "Analyzing: " <> T.unpack (Dhall.targetName target)
    analysisResult <- DICE.analyze target
    
    if not (null (DICE.arErrors analysisResult))
      then do
        hPutStrLn stderr "Resolution failed:"
        forM_ (DICE.arErrors analysisResult) $ \e ->
          hPutStrLn stderr $ "  " <> T.unpack e
        exitFailure
      else do
        let graph = DICE.arGraph analysisResult
        putStrLn $ "Executing " <> show (length $ DICE.agActions graph) <> " action(s)..."
        putStrLn ""
        

        
        execResult <- DICE.executeGraph graph
        
        putStrLn $ "Cache hits: " <> show (DICE.erCacheHits execResult)
        putStrLn $ "Executed:   " <> show (DICE.erExecuted execResult)
        
        if null (DICE.erFailed execResult)
          then do
            putStrLn ""
            putStrLn "Outputs:"
            forM_ (Map.toList $ DICE.erOutputs execResult) $ \(key, paths) ->
              forM_ paths $ \p ->
                putStrLn $ "  " <> T.unpack p
          else do
            hPutStrLn stderr ""
            hPutStrLn stderr "Failures:"
            forM_ (DICE.erFailed execResult) $ \(key, err) ->
              hPutStrLn stderr $ "  " <> T.unpack (DICE.unActionKey key) <> ": " <> T.unpack err
            exitFailure

-- | Trace command - intercept build system via strace
cmdTrace :: [String] -> IO ()
cmdTrace args = case args of
  [] -> do
    hPutStrLn stderr "Usage: armitage trace [options] -- <build command>"
    hPutStrLn stderr ""
    hPutStrLn stderr "Options:"
    hPutStrLn stderr "  -o <file>    Output Dhall file (default: stdout)"
    hPutStrLn stderr "  -v           Verbose mode"
    hPutStrLn stderr ""
    hPutStrLn stderr "Example:"
    hPutStrLn stderr "  armitage trace -- cmake --build build/"
    hPutStrLn stderr "  armitage trace -o BUILD.dhall -- make -j8"
    exitFailure
  _ -> do
    let (opts, cmd) = parseTraceArgs args
        cfg = Trace.defaultTraceConfig { Trace.tcVerbose = "-v" `elem` opts }
        outputFile = parseOutputFile opts
    
    when (null cmd) $ do
      hPutStrLn stderr "Error: No build command specified after --"
      exitFailure
    
    putStrLn $ "Tracing: " <> unwords cmd
    putStrLn "Running build under strace..."
    putStrLn ""
    
    result <- Trace.traceCommand cfg cmd
    case result of
      Left err -> do
        hPutStrLn stderr $ "Trace failed: " <> T.unpack err
        exitFailure
      Right traceOutput -> do
        let (compiles, links) = Trace.parseStrace cfg traceOutput
        putStrLn $ "Captured:"
        putStrLn $ "  " <> show (length compiles) <> " compile call(s)"
        putStrLn $ "  " <> show (length links) <> " link call(s)"
        putStrLn ""
        
        when (null compiles && null links) $ do
          hPutStrLn stderr "Warning: No compiler/linker calls detected"
          hPutStrLn stderr "Make sure the build actually compiles something"
        
        -- Analyze into build graph
        buildGraph <- Trace.analyzeTrace cfg (compiles, links)
        
        putStrLn $ "Extracted " <> show (length $ Trace.bgTargets buildGraph) <> " target(s)"
        forM_ (Trace.bgTargets buildGraph) $ \t ->
          putStrLn $ "  - " <> T.unpack (Trace.tName t)
        putStrLn ""
        
        -- Generate Dhall
        let dhall = Trace.toDhall buildGraph
        case outputFile of
          Nothing -> do
            putStrLn "Generated Dhall:"
            putStrLn "────────────────────────────────────────"
            TIO.putStrLn dhall
          Just path -> do
            Trace.toDhallFile path buildGraph
            putStrLn $ "Wrote: " <> path
  where
    parseOutputFile [] = Nothing
    parseOutputFile ("-o":f:_) = Just f
    parseOutputFile (_:rest) = parseOutputFile rest

-- | Parse trace args, splitting on --
parseTraceArgs :: [String] -> ([String], [String])
parseTraceArgs args = 
  let (before, after) = break (== "--") args
  in (before, drop 1 after)  -- drop the "--"

usage :: IO ()
usage = do
  prog <- getProgName
  putStrLn $ "Usage: " <> prog <> " <command> [options]"
  putStrLn ""
  putStrLn "Daemon-free Nix operations"
  putStrLn ""
  putStrLn "Commands:"
  putStrLn "  build <drv>        Build derivation without daemon"
  putStrLn "  build-dhall <file> Build from Dhall target file"
  putStrLn "  analyze <file>     Analyze deps and build action graph"
  putStrLn "  run <file>         Analyze and execute build"
  putStrLn "  trace -- <cmd>     Trace build, extract Dhall targets"
  putStrLn "  proxy              Run witness proxy"
  putStrLn "  store <cmd>        Store operations"
  putStrLn "  cas <cmd>          Content-addressed storage"
  putStrLn ""
  putStrLn "The daemon is hostile infrastructure. armitage routes around it."
