{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- |
-- render - CLI for nix-compile type inference
--
-- Usage:
--   render parse <script>       Parse and show facts
--   render infer <script>       Infer types and show schema
--   render check <script>       Check for policy violations
--   render lint <script>        Check for forbidden constructs
--   render emit <script> [fmt]  Generate emit-config function
--   render nix <file.nix>       Check embedded bash in Nix files
module Main where

import Data.Aeson (encode)
import qualified Data.ByteString.Lazy.Char8 as BL
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)
import NixCompile
import NixCompile.Bash.Parse (parseBash)
import NixCompile.Emit.Config (emitConfigFunction)
import NixCompile.Lint.Forbidden (findViolations, formatViolations)
import qualified NixCompile.Nix.Infer
import qualified NixCompile.Nix.Parse as Nix
import qualified NixCompile.Nix.Format as NixFmt
import qualified NixCompile.Nix.Flake as Flake
import qualified NixCompile.Nix.Module as Mod
import qualified NixCompile.Nix.Layout as Layout
import qualified NixCompile.Nix.Lint as Lint
import qualified NixCompile.Nix.Scope as Scope
import qualified NixCompile.Nix.Types
import NixCompile.Types (Loc(..), Span(..))
import qualified NixCompile.Nix.Pretty as Pretty
import qualified Data.Map.Strict as Map
import Control.Monad (forM, forM_, mapM)
import Control.Concurrent.Async (mapConcurrently)
import Control.Concurrent.MVar (newMVar, withMVar)
import System.Directory (doesDirectoryExist, listDirectory)
import System.FilePath ((</>), takeExtension, makeRelative, takeDirectory)
import Control.Exception (catch, evaluate, SomeException)

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["lint", file] -> cmdLint file
    ["check", file] -> cmdCheck file
    ["infer", file] -> cmdInfer file
    ["parse", file] -> cmdParse file
    ["emit", file] -> cmdEmit file
    ["nix", file] -> cmdNix file
    ["fmt", file] -> cmdFmt file
    ["typecheck", path] -> cmdTypeCheck path
    ["flake"] -> cmdFlake "."
    ["flake", dir] -> cmdFlake dir
    ["graph"] -> cmdGraph "." False
    ["graph", dir] -> cmdGraph dir False
    ["graph", "--dot"] -> cmdGraph "." True
    ["graph", "--dot", dir] -> cmdGraph dir True
    ["graph", dir, "--dot"] -> cmdGraph dir True
    ["scope", file] -> cmdScope file
    ["scope", "--json", file] -> cmdScopeJSON file
    ["scope", "--dhall", file] -> cmdScopeDhall file
    ["--help"] -> usage
    ["-h"] -> usage
    [] -> usage
    _ -> do
      putStrLn $ "Unknown command: " ++ unwords args
      usage
      exitFailure

usage :: IO ()
usage = do
  putStrLn "render - typed shell scripts for Nix"
  putStrLn ""
  putStrLn "Usage:"
  putStrLn "  render lint <script.sh>    Check for forbidden constructs (heredocs, eval, etc)"
  putStrLn "  render check <script.sh>   Full check (lint + bare commands + types)"
  putStrLn "  render infer <script.sh>   Infer types and show schema (JSON)"
  putStrLn "  render parse <script.sh>   Parse and show extracted facts"
  putStrLn "  render emit <script.sh>    Generate emit-config Dhall function"
  putStrLn "  render nix <file.nix>      Check embedded bash in Nix files"
  putStrLn "  render fmt <file.nix>      Add type annotations to Nix file"
  putStrLn "  render typecheck <path>    Recursively infer and check types for all Nix files"
  putStrLn "  render flake [dir]         Analyze a flake"
  putStrLn "  render graph [--dot] [dir] Show module dependency graph (exits 1 on violations)"
  putStrLn "  render scope <file.nix>    Show scope graph (declarations, references, edges)"
  putStrLn "  render scope --json <file> Emit scope graph as JSON (for zeitschrift)"
  putStrLn "  render scope --dhall <file> Emit scope graph as Dhall (for zeitschrift)"
  putStrLn ""
  putStrLn "Forbidden bash constructs (no escape hatch):"
  putStrLn "  - heredocs (<<, <<-)"
  putStrLn "  - here-strings (<<<)"
  putStrLn "  - eval"
  putStrLn "  - backticks (`cmd`)"
  putStrLn "  - bare commands (use store paths)"
  putStrLn ""
  putStrLn "Forbidden Nix constructs (no escape hatch):"
  putStrLn "  - with expr;  (obscures scope, breaks tooling)"
  putStrLn "  - rec { }     (enables non-termination, breaks analysis)"
  putStrLn "  - \"str\".attr  (breaks hnix parser)"
  putStrLn ""
  putStrLn "Examples:"
  putStrLn "  render lint ./deploy.sh"
  putStrLn "  render check ./scripts/*.sh"
  putStrLn "  render infer ./deploy.sh | jq '.env'"
  putStrLn "  render nix ./default.nix"

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

-- | Lint for forbidden constructs only (heredocs, eval, backticks)
cmdLint :: FilePath -> IO ()
cmdLint file = do
  src <- TIO.readFile file
  case parseBash src of
    Left err -> do
      TIO.putStrLn $ "Parse error: " <> err
      exitFailure
    Right ast -> do
      let violations = findViolations ast
      if null violations
        then do
          putStrLn $ file ++ ": OK (no forbidden constructs)"
          exitSuccess
        else do
          TIO.putStr $ formatViolations violations
          putStrLn $ "\n" ++ show (length violations) ++ " error(s) in " ++ file
          exitFailure

-- | Full check: lint + bare commands + type inference
cmdCheck :: FilePath -> IO ()
cmdCheck file = do
  src <- TIO.readFile file
  -- First check for forbidden constructs
  case parseBash src of
    Left err -> do
      TIO.putStrLn $ "Parse error: " <> err
      exitFailure
    Right ast -> do
      let violations = findViolations ast
      unless (null violations) $ do
        TIO.putStr $ formatViolations violations
        putStrLn ""
      
      -- Then do type inference and check bare commands
      case parseScript src of
        Left err -> do
          TIO.putStrLn $ "Type error: " <> err
          exitFailure
        Right script -> do
          let schema = scriptSchema script
          let bareCount = length (schemaBareCommands schema)
          let dynCount = length (schemaDynamicCommands schema)
          let violationCount = length violations
          
          -- Report bare commands
          when (bareCount > 0) $ do
            putStrLn "Bare commands (must use store paths):"
            mapM_ (\cmd -> putStrLn $ "  " ++ T.unpack cmd) (schemaBareCommands schema)
          
          when (dynCount > 0) $ do
            putStrLn "Dynamic commands (cannot analyze):"
            mapM_ (\cmd -> putStrLn $ "  $" ++ T.unpack cmd) (schemaDynamicCommands schema)
          
          let totalErrors = violationCount + bareCount + dynCount
          if totalErrors > 0
            then do
              putStrLn $ "\n" ++ show totalErrors ++ " error(s) in " ++ file
              exitFailure
            else do
              putStrLn $ file ++ ": OK"
              exitSuccess
  where
    unless cond action = if cond then return () else action
    when cond action = if cond then action else return ()

cmdEmit :: FilePath -> IO ()
cmdEmit file = do
  result <- parseScriptFile file
  case result of
    Left err -> do
      TIO.putStrLn $ "Error: " <> err
      exitFailure
    Right script -> do
      TIO.putStr $ emitConfigFunction (scriptSchema script)

-- | Check embedded bash scripts in Nix files
cmdNix :: FilePath -> IO ()
cmdNix file = do
  result <- Nix.extractBashScripts file
  case result of
    Left err -> do
      TIO.putStrLn $ "Parse error: " <> err
      exitFailure
    Right scripts -> do
      putStrLn $ "Found " ++ show (length scripts) ++ " shell scripts in " ++ file
      totalErrors <- sum <$> mapM checkScript scripts
      if totalErrors > 0
        then do
          putStrLn $ "\n" ++ show totalErrors ++ " total error(s)"
          exitFailure
        else do
          putStrLn $ file ++ ": OK"
          exitSuccess
  where
    checkScript :: Nix.BashScript -> IO Int
    checkScript bs = do
      putStrLn $ "\n=== " ++ T.unpack (Nix.bsName bs) ++ " ==="
      -- Parse and check the bash content
      case parseBash (Nix.bsContent bs) of
        Left err -> do
          TIO.putStrLn $ "  Parse error: " <> err
          return 1
        Right ast -> do
          -- Check for forbidden constructs
          let violations = findViolations ast
          unless (null violations) $ do
            TIO.putStr $ formatViolations violations
          
          -- Check for non-store-path interpolations
          let badInterps = filter (not . Nix.intIsStorePath) (Nix.bsInterpolations bs)
          unless (null badInterps) $ do
            putStrLn "  Non-store-path interpolations (may need verification):"
            mapM_ (\i -> putStrLn $ "    ${" ++ T.unpack (Nix.intExpr i) ++ "}") badInterps
          
          -- Type inference
          case parseScript (Nix.bsContent bs) of
            Left err -> do
              TIO.putStrLn $ "  Type error: " <> err
              return (length violations + 1)
            Right script -> do
              let schema = scriptSchema script
              let bareCount = length (schemaBareCommands schema)
              
              when (bareCount > 0) $ do
                putStrLn "  Bare commands (must use store paths):"
                mapM_ (\cmd -> putStrLn $ "    " ++ T.unpack cmd) (schemaBareCommands schema)
              
              let errorCount = length violations + bareCount
              if errorCount == 0
                then putStrLn "  OK"
                else putStrLn $ "  " ++ show errorCount ++ " error(s)"
              return errorCount
    
    unless cond action = if cond then return () else action
    when cond action = if cond then action else return ()

-- | Recursively type check a directory or single file in parallel
cmdTypeCheck :: FilePath -> IO ()
cmdTypeCheck path = do
  isDir <- doesDirectoryExist path
  files <- if isDir 
           then findAllNixFiles path
           else return [path]
  
  putStrLn $ "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  putStrLn $ "  nix-compile typecheck"
  putStrLn $ "  " ++ show (length files) ++ " files"
  putStrLn $ "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  putStrLn ""
  
  -- Lock for synchronized output
  outLock <- newMVar ()
  let log msg = withMVar outLock $ \_ -> TIO.putStrLn msg
  
  results <- mapConcurrently (checkFile log) files
  let failures = filter not results
  let successCount = length files - length failures
  
  putStrLn ""
  putStrLn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  if null failures
    then do
      putStrLn $ "  ✓ All " ++ show successCount ++ " files passed"
      putStrLn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      exitSuccess
    else do
      putStrLn $ "  ✓ " ++ show successCount ++ " passed"
      putStrLn $ "  ✗ " ++ show (length failures) ++ " failed"
      putStrLn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      exitFailure
  where
    findAllNixFiles :: FilePath -> IO [FilePath]
    findAllNixFiles dir = do
      entries <- listDirectory dir
      paths <- forM entries $ \entry -> do
        let fullPath = dir </> entry
        isD <- doesDirectoryExist fullPath
        if isD
          then findAllNixFiles fullPath
          else return [fullPath | takeExtension fullPath == ".nix"]
      return (concat paths)

    checkFile :: (T.Text -> IO ()) -> FilePath -> IO Bool
    checkFile log file = do
      -- Catch all exceptions (including pure ones like error calls)
      result <- try $ do
        res <- Nix.parseNixFile file
        case res of
          Left err -> return $ Left ("parse error\n  " <> err)
          Right expr -> case NixCompile.Nix.Infer.inferExpr expr of
            Left err -> return $ Left err
            Right (t, _) -> return $ Right (NixCompile.Nix.Types.prettyType t)
            
      case result of
        Left (e :: SomeException) -> do
          log $ "✗ " <> T.pack file <> "\n  internal error: " <> T.pack (show e)
          return False
        Right (Left err) -> do
          log $ "✗ " <> T.pack file <> "\n  " <> err
          return False
        Right (Right t) -> do
          log $ "✓ " <> T.pack file <> "\n  " <> t
          return True

    try :: IO a -> IO (Either SomeException a)
    try act = catch (Right <$> act) (\e -> return (Left e))

-- | Format a Nix file with type annotations
cmdFmt :: FilePath -> IO ()
cmdFmt file = do
  result <- NixFmt.formatFile file
  case result of
    Left err -> do
      TIO.putStrLn $ "Error: " <> err
      exitFailure
    Right formatted -> do
      TIO.putStr formatted

-- | Analyze a flake
cmdFlake :: FilePath -> IO ()
cmdFlake dir = do
  result <- Flake.parseFlakeDir dir
  case result of
    Left err -> do
      TIO.putStrLn $ "Error: " <> err
      exitFailure
    Right flake -> do
      putStrLn "=== Flake ==="
      putStrLn $ "Path: " ++ Flake.flakePath flake
      TIO.putStrLn $ "Description: " <> maybe "(none)" id (Flake.flakeDescription flake)
      
      putStrLn "\n=== Inputs ==="
      mapM_ printInput (Map.toList $ Flake.flakeInputs flake)
      
      putStrLn "\n=== Inferred Type ==="
      let types = Flake.inferFlake flake
      TIO.putStrLn $ "outputs : " <> prettyType (Flake.ftOutputsType types)
  where
    printInput (name, input) = do
      TIO.putStr $ "  " <> name <> " : FlakeInput"
      case Flake.inputUrl input of
        Just url -> TIO.putStrLn $ " = \"" <> url <> "\""
        Nothing -> case Flake.inputFollows input of
          Just follows -> TIO.putStrLn $ " (follows " <> follows <> ")"
          Nothing -> putStrLn ""
    
    prettyType = NixCompile.Nix.Types.prettyType

-- | Show module dependency graph
cmdGraph :: FilePath -> Bool -> IO ()
cmdGraph dir asDot = do
  result <- Mod.buildModuleGraphFromFlake dir
  case result of
    Left err -> do
      TIO.putStrLn $ "Error: " <> err
      exitFailure
    Right graph -> do
      let rootDir = takeDirectory (Mod.mgRoot graph)
      if asDot
        then printDot rootDir graph
        else do
          printGraph rootDir graph
          -- Exit with failure if there are any violations
          if Mod.hasViolations graph
            then exitFailure
            else exitSuccess
  where
    printGraph :: FilePath -> Mod.ModuleGraph -> IO ()
    printGraph rootDir graph = do
      putStrLn "=== Module Graph ==="
      putStrLn $ "Root: " ++ makeRelative rootDir (Mod.mgRoot graph)
      putStrLn $ "Modules: " ++ show (Map.size (Mod.mgModules graph))
      
      let parseFailures = Mod.mgFailures graph
      let lintFailures = Mod.mgLintFailures graph
      let layoutFailures = Mod.mgLayoutFailures graph
      let lintViolationCount = sum (map (length . Mod.lfViolations) lintFailures)
      let layoutViolationCount = sum (map (length . Mod.layViolations) layoutFailures)
      
      if null parseFailures && null lintFailures && null layoutFailures
        then putStrLn ""
        else do
          if not (null parseFailures)
            then do
              putStrLn $ "Parse failures: " ++ show (length parseFailures)
              putStrLn ""
              putStrLn "=== Parse Failures (banned syntax) ==="
              mapM_ (printParseFailure rootDir) parseFailures
              putStrLn ""
            else return ()
          
          if not (null lintFailures)
            then do
              putStrLn $ "Lint violations: " ++ show lintViolationCount ++ " in " ++ show (length lintFailures) ++ " files"
              putStrLn ""
              putStrLn "=== Lint Failures (with/rec banned) ==="
              mapM_ (printLintFailure rootDir) lintFailures
              putStrLn ""
            else return ()
          
          if not (null layoutFailures)
            then do
              putStrLn $ "Layout violations: " ++ show layoutViolationCount ++ " in " ++ show (length layoutFailures) ++ " files"
              putStrLn ""
              putStrLn "=== Layout Failures (directory structure) ==="
              mapM_ (printLayoutFailure rootDir) layoutFailures
              putStrLn ""
            else return ()
      
      putStrLn "=== Topological Order (dependencies first) ==="
      mapM_ (\p -> putStrLn $ "  " ++ makeRelative rootDir p) (Mod.mgOrder graph)
      putStrLn ""
      
      putStrLn "=== Import Graph ==="
      mapM_ (printModuleImports rootDir) (Map.elems (Mod.mgModules graph))
      
      putStrLn ""
      putStrLn "=== Module Types ==="
      mapM_ (printModuleType rootDir graph) (Mod.mgOrder graph)
    
    printParseFailure :: FilePath -> Mod.ParseFailure -> IO ()
    printParseFailure rootDir pf = do
      let path = makeRelative rootDir (Mod.pfPath pf)
      TIO.putStrLn $ "  " <> T.pack path <> ":"
      -- Extract just the first line of the error (location info)
      let errLines = T.lines (Mod.pfError pf)
      case errLines of
        (l:_) -> TIO.putStrLn $ "    " <> l
        [] -> return ()
    
    printLintFailure :: FilePath -> Mod.LintFailure -> IO ()
    printLintFailure rootDir lf = do
      let path = makeRelative rootDir (Mod.lfPath lf)
      TIO.putStrLn $ "  " <> T.pack path <> ":"
      mapM_ printViolation (Mod.lfViolations lf)
    
    printViolation :: Lint.NixViolation -> IO ()
    printViolation v = do
      let loc = Lint.nvSpan v
      let code = case Lint.nvType v of
            Lint.VWith -> "ALEPH-N001"
            Lint.VRec -> "ALEPH-N002"
      TIO.putStrLn $ "    " <> T.pack (show (locLine (spanStart loc))) <> ":" <>
                     T.pack (show (locCol (spanStart loc))) <> " " <> code <> ": " <>
                     Lint.nvContext v
    
    locLine (Loc l _) = l
    locCol (Loc _ c) = c
    spanStart (Span s _ _) = s
    
    printLayoutFailure :: FilePath -> Mod.LayoutFailure -> IO ()
    printLayoutFailure rootDir lf = do
      let path = makeRelative rootDir (Mod.layPath lf)
      TIO.putStrLn $ "  " <> T.pack path <> ":"
      mapM_ printLayoutViolation (Mod.layViolations lf)
    
    printLayoutViolation :: Layout.LayoutViolation -> IO ()
    printLayoutViolation v = do
      let code = case Layout.lvCode v of
            Layout.L001 -> "ALEPH-L001"
            Layout.L002 -> "ALEPH-L002"
            Layout.L003 -> "ALEPH-L003"
            Layout.L004 -> "ALEPH-L004"
            Layout.L005 -> "ALEPH-L005"
      case Layout.lvSpan v of
        Just loc -> 
          TIO.putStrLn $ "    " <> T.pack (show (locLine (spanStart loc))) <> ":" <>
                         T.pack (show (locCol (spanStart loc))) <> " " <> code <> ": " <>
                         Layout.lvMessage v
        Nothing ->
          TIO.putStrLn $ "    " <> code <> ": " <> Layout.lvMessage v
    
    printModuleImports :: FilePath -> Mod.Module -> IO ()
    printModuleImports rootDir m = do
      let path = makeRelative rootDir (Mod.modPath m)
      let imports = Mod.modImports m
      if null imports
        then return ()
        else do
          putStrLn $ path ++ ":"
          mapM_ (\imp -> putStrLn $ "  -> " ++ makeRelative rootDir (Mod.impPath imp)) imports
    
    printModuleType :: FilePath -> Mod.ModuleGraph -> FilePath -> IO ()
    printModuleType rootDir graph path =
      case Map.lookup path (Mod.mgModules graph) of
        Nothing -> return ()
        Just m -> do
          let relPath = makeRelative rootDir path
          TIO.putStrLn $ T.pack relPath <> " : " <> NixCompile.Nix.Types.prettyType (Mod.modType m)
    
    printDot :: FilePath -> Mod.ModuleGraph -> IO ()
    printDot rootDir graph = do
      putStrLn "digraph modules {"
      putStrLn "  rankdir=LR;"
      putStrLn "  node [shape=box];"
      putStrLn ""
      mapM_ (printDotEdges rootDir) (Map.elems (Mod.mgModules graph))
      putStrLn "}"
    
    printDotEdges :: FilePath -> Mod.Module -> IO ()
    printDotEdges rootDir m = do
      let path = makeRelative rootDir (Mod.modPath m)
      mapM_ (\imp -> do
        let impPath = makeRelative rootDir (Mod.impPath imp)
        putStrLn $ "  \"" ++ path ++ "\" -> \"" ++ impPath ++ "\";") (Mod.modImports m)

-- | Show scope graph for a Nix file
cmdScope :: FilePath -> IO ()
cmdScope file = do
  result <- Nix.parseNixFile file
  case result of
    Left err -> do
      TIO.putStrLn $ "Parse error: " <> err
      exitFailure
    Right expr -> do
      let sg = Scope.fromNixFile file expr
      printScopeGraph sg

-- | Emit scope graph as JSON (for zeitschrift)
cmdScopeJSON :: FilePath -> IO ()
cmdScopeJSON file = do
  result <- Nix.parseNixFile file
  case result of
    Left err -> do
      TIO.putStrLn $ "Parse error: " <> err
      exitFailure
    Right expr -> do
      let sg = Scope.fromNixFile file expr
      BL.putStrLn $ encode sg

-- | Emit scope graph as Dhall (for zeitschrift)
cmdScopeDhall :: FilePath -> IO ()
cmdScopeDhall file = do
  result <- Nix.parseNixFile file
  case result of
    Left err -> do
      TIO.putStrLn $ "Parse error: " <> err
      exitFailure
    Right expr -> do
      let sg = Scope.fromNixFile file expr
      TIO.putStrLn $ Scope.toDhall sg

printScopeGraph :: Scope.ScopeGraph -> IO ()
printScopeGraph sg = do
  putStrLn "=== Scope Graph ==="
  putStrLn $ "File: " ++ maybe "(none)" id (Scope.sgFile sg)
  putStrLn $ "Scopes: " ++ show (Map.size (Scope.sgScopes sg))
  putStrLn ""
  
  -- Print scopes with their contents
  forM_ (Map.elems (Scope.sgScopes sg)) $ \scope -> do
    putStrLn $ "Scope " ++ show (Scope.unScopeId (Scope.scopeId scope)) ++ 
               " (" ++ show (Scope.scopeKind scope) ++ "):"
    
    -- Declarations
    let decls = Scope.scopeDeclarations scope
    unless (null decls) $ do
      putStrLn "  Declarations:"
      forM_ decls $ \d -> do
        TIO.putStrLn $ "    " <> Scope.declName d <> 
                       maybe "" (\t -> " : " <> t) (Scope.declType d)
    
    -- References
    let refs = Scope.scopeReferences scope
    unless (null refs) $ do
      putStrLn "  References:"
      forM_ refs $ \r -> do
        TIO.putStrLn $ "    " <> Scope.refName r <> " (" <> T.pack (show (Scope.refKind r)) <> ")"
    
    -- Edges
    let edges = Scope.scopeEdges scope
    unless (null edges) $ do
      putStrLn "  Edges:"
      forM_ edges $ \e -> do
        putStrLn $ "    -> " ++ show (Scope.unScopeId (Scope.edgeTarget e)) ++
                   " (" ++ show (Scope.edgeLabel e) ++ ")"
    
    putStrLn ""
  
  -- Resolution summary
  case Scope.resolveAll sg of
    Left errors -> do
      putStrLn $ "=== Unresolved References (" ++ show (length errors) ++ ") ==="
      forM_ errors $ \case
        Scope.Unresolved ref -> TIO.putStrLn $ "  " <> Scope.refName ref
        Scope.Ambiguous ref _ -> TIO.putStrLn $ "  " <> Scope.refName ref <> " (ambiguous)"
    Right resolved -> do
      putStrLn $ "=== All " ++ show (length resolved) ++ " references resolved ==="
  where
    unless cond action = if cond then return () else action
    forM_ = flip mapM_
