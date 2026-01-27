{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Armitage.LSP
Description : Trace-based language server with graceful degradation

The All Seeing Eye.

Architecture:
  1. Proxy to real LSPs (rust-analyzer, clangd, hls) when available
  2. Fall back to tree-sitter + trace when they're slow/crashed/missing
  3. Diagnostics always come from real compiler (via trace)
  4. Never fail completely - degrade gracefully

The trace gives us ground truth:
  - execve: what compiler ran, with what flags
  - open(O_RDONLY): what files were inputs
  - open(O_WRONLY): what files were outputs  
  - stderr: real diagnostics

Tree-sitter gives us syntax:
  - All symbols in all files
  - Scope information
  - Structural navigation

Combined: an LSP that's always correct about what it knows,
and honest about what it doesn't.

Usage:
  armitage lsp              # Start LSP server
  armitage lsp --port 9999  # Specific port
  armitage lsp --stdio      # Stdio mode (for editors)
-}
module Armitage.LSP
  ( -- * Server
    startLSP
  , LSPConfig (..)
  , defaultLSPConfig
  
    -- * Backends
  , Backend (..)
  , BackendStatus (..)
  , selectBackend
  
    -- * Fallback
  , FallbackState (..)
  , initFallback
  
    -- * Compile Commands (from shims)
  , CompileCommand (..)
  , generateCompileCommands
  , writeCompileCommands
  
    -- * Protocol Types
  , Request (..)
  , Response (..)
  , Position (..)
  , Range (..)
  , Location (..)
  , Diagnostic (..)
  , DiagnosticSeverity (..)
  , Symbol (..)
  , SymbolKind (..)
  , CompletionItem (..)
  ) where

import Control.Concurrent (ThreadId, forkIO, killThread)
import Control.Concurrent.MVar
import Control.Concurrent.STM
import Control.Exception (SomeException, catch, try)
import Control.Monad (forever, forM, forM_, when, void)
import Data.Aeson (ToJSON (..), FromJSON (..), (.=), (.:), (.:?))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AesonKey
import qualified Data.Aeson.KeyMap as AesonKM
import Data.Function ((&))
import Data.ByteString (ByteString)
import qualified Data.ByteString.Char8 as B8
import qualified Data.ByteString.Lazy as BL
import Data.IORef
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe, catMaybes, mapMaybe)
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Data.Time.Clock (UTCTime, getCurrentTime, diffUTCTime)
import GHC.Generics (Generic)
import System.Exit (ExitCode (..))
import System.FilePath (takeExtension, takeFileName)
import System.IO (Handle, hFlush, hGetLine, hPutStr, stdin, stdout, stderr)
import System.Process (ProcessHandle, CreateProcess (..), StdStream (..), 
                       createProcess, proc, terminateProcess, waitForProcess,
                       getProcessExitCode)
import System.Timeout (timeout)

import qualified Armitage.Shim as Shim
import qualified Armitage.Trace as Trace

-- -----------------------------------------------------------------------------
-- Configuration
-- -----------------------------------------------------------------------------

data LSPConfig = LSPConfig
  { lcBackendTimeout :: Int           -- ^ Timeout for real LSP (microseconds)
  , lcFallbackEnabled :: Bool         -- ^ Enable fallback mode
  , lcTraceCacheDir :: FilePath       -- ^ Where to cache trace data
  , lcTreeSitterParsers :: FilePath   -- ^ Tree-sitter parser directory
  , lcKnownLSPs :: Map Text Text      -- ^ Language -> LSP command
  , lcVerbose :: Bool
  }
  deriving stock (Show, Eq, Generic)

defaultLSPConfig :: LSPConfig
defaultLSPConfig = LSPConfig
  { lcBackendTimeout = 100000  -- 100ms
  , lcFallbackEnabled = True
  , lcTraceCacheDir = ".armitage/trace"
  , lcTreeSitterParsers = ".armitage/parsers"
  , lcKnownLSPs = Map.fromList
      [ ("rust", "rust-analyzer")
      , ("c", "clangd")
      , ("cpp", "clangd")
      , ("haskell", "haskell-language-server-wrapper --lsp")
      , ("python", "pylsp")
      , ("typescript", "typescript-language-server --stdio")
      , ("javascript", "typescript-language-server --stdio")
      , ("go", "gopls")
      , ("zig", "zls")
      , ("nix", "nil")
      ]
  , lcVerbose = False
  }

-- -----------------------------------------------------------------------------
-- Protocol Types (subset of LSP spec)
-- -----------------------------------------------------------------------------

data Position = Position
  { posLine :: !Int
  , posCharacter :: !Int
  }
  deriving stock (Show, Eq, Ord, Generic)
  deriving anyclass (ToJSON, FromJSON)

data Range = Range
  { rangeStart :: !Position
  , rangeEnd :: !Position
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON, FromJSON)

data Location = Location
  { locUri :: !Text
  , locRange :: !Range
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON, FromJSON)

data DiagnosticSeverity = Error | Warning | Information | Hint
  deriving stock (Show, Eq, Ord, Generic)
  deriving anyclass (ToJSON, FromJSON)

data Diagnostic = Diagnostic
  { diagRange :: !Range
  , diagSeverity :: !(Maybe DiagnosticSeverity)
  , diagCode :: !(Maybe Text)
  , diagSource :: !(Maybe Text)
  , diagMessage :: !Text
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON, FromJSON)

data SymbolKind
  = SkFile | SkModule | SkNamespace | SkPackage | SkClass | SkMethod
  | SkProperty | SkField | SkConstructor | SkEnum | SkInterface
  | SkFunction | SkVariable | SkConstant | SkString | SkNumber
  | SkBoolean | SkArray | SkObject | SkKey | SkNull | SkEnumMember
  | SkStruct | SkEvent | SkOperator | SkTypeParameter
  deriving stock (Show, Eq, Ord, Generic)
  deriving anyclass (ToJSON, FromJSON)

data Symbol = Symbol
  { symName :: !Text
  , symKind :: !SymbolKind
  , symLocation :: !Location
  , symContainerName :: !(Maybe Text)
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON, FromJSON)

data CompletionItem = CompletionItem
  { ciLabel :: !Text
  , ciKind :: !(Maybe SymbolKind)
  , ciDetail :: !(Maybe Text)
  , ciDocumentation :: !(Maybe Text)
  , ciInsertText :: !(Maybe Text)
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON, FromJSON)

-- -----------------------------------------------------------------------------
-- Request/Response
-- -----------------------------------------------------------------------------

data Request
  = Initialize Aeson.Value
  | Shutdown
  | TextDocumentDidOpen Text Text      -- uri, content
  | TextDocumentDidChange Text Text    -- uri, content
  | TextDocumentDidClose Text          -- uri
  | TextDocumentCompletion Text Position
  | TextDocumentHover Text Position
  | TextDocumentDefinition Text Position
  | TextDocumentReferences Text Position
  | TextDocumentSymbol Text
  | TextDocumentDiagnostics Text
  | WorkspaceSymbol Text               -- query
  | UnknownRequest Text Aeson.Value
  deriving stock (Show, Eq, Generic)

data Response
  = InitializeResult Aeson.Value
  | ShutdownResult
  | CompletionResult [CompletionItem]
  | HoverResult (Maybe Text)
  | DefinitionResult [Location]
  | ReferencesResult [Location]
  | SymbolResult [Symbol]
  | DiagnosticsResult [Diagnostic]
  | ErrorResponse Int Text
  | NullResult
  deriving stock (Show, Eq, Generic)

-- -----------------------------------------------------------------------------
-- Backend Management
-- -----------------------------------------------------------------------------

data Backend
  = RealLSP 
      { blspProcess :: !ProcessHandle
      , blspStdin :: !Handle
      , blspStdout :: !Handle
      , blspLanguage :: !Text
      }
  | Fallback

data BackendStatus
  = BackendHealthy
  | BackendSlow          -- Responded, but too slow
  | BackendCrashed       -- Process died
  | BackendUnavailable   -- Not installed
  deriving stock (Show, Eq, Generic)

-- | Backend state per language
data BackendState = BackendState
  { bsBackend :: !(Maybe Backend)
  , bsStatus :: !BackendStatus
  , bsLastResponse :: !(Maybe UTCTime)
  , bsSlowCount :: !Int          -- How many times it was slow
  , bsCrashCount :: !Int         -- How many times it crashed
  }

-- | Select backend for a language, spawning if needed
selectBackend :: LSPConfig -> TVar (Map Text BackendState) -> Text -> IO Backend
selectBackend cfg stateVar lang = do
  states <- readTVarIO stateVar
  case Map.lookup lang states of
    Just bs | bsStatus bs == BackendHealthy
            , Just backend <- bsBackend bs -> pure backend
    
    Just bs | bsCrashCount bs > 3 -> pure Fallback  -- Give up after 3 crashes
    
    _ -> trySpawnBackend cfg stateVar lang

-- | Try to spawn a real LSP backend
trySpawnBackend :: LSPConfig -> TVar (Map Text BackendState) -> Text -> IO Backend
trySpawnBackend cfg stateVar lang =
  case Map.lookup lang (lcKnownLSPs cfg) of
    Nothing -> do
      updateBackendState stateVar lang $ \bs -> 
        bs { bsStatus = BackendUnavailable }
      pure Fallback
    
    Just cmd -> do
      result <- try @SomeException $ spawnLSP cmd
      case result of
        Left _ -> do
          updateBackendState stateVar lang $ \bs ->
            bs { bsStatus = BackendUnavailable }
          pure Fallback
        
        Right (process, stdin', stdout') -> do
          let backend = RealLSP process stdin' stdout' lang
          updateBackendState stateVar lang $ \bs ->
            bs { bsBackend = Just backend
               , bsStatus = BackendHealthy
               }
          pure backend

-- | Spawn an LSP process
spawnLSP :: Text -> IO (ProcessHandle, Handle, Handle)
spawnLSP cmd = do
  let parts = T.words cmd
      exe = T.unpack (head parts)
      args = map T.unpack (tail parts)
  
  (Just stdin', Just stdout', _, process) <- createProcess (proc exe args)
    { std_in = CreatePipe
    , std_out = CreatePipe
    , std_err = CreatePipe
    }
  
  pure (process, stdin', stdout')

-- | Update backend state atomically
updateBackendState :: TVar (Map Text BackendState) -> Text -> (BackendState -> BackendState) -> IO ()
updateBackendState stateVar lang f = atomically $ do
  states <- readTVar stateVar
  let current = Map.findWithDefault emptyBackendState lang states
      updated = f current
  writeTVar stateVar (Map.insert lang updated states)

emptyBackendState :: BackendState
emptyBackendState = BackendState
  { bsBackend = Nothing
  , bsStatus = BackendUnavailable
  , bsLastResponse = Nothing
  , bsSlowCount = 0
  , bsCrashCount = 0
  }

-- -----------------------------------------------------------------------------
-- Fallback State (tree-sitter + trace)
-- -----------------------------------------------------------------------------

data FallbackState = FallbackState
  { fsFiles :: !(TVar (Map Text FileState))    -- Parsed files
  , fsGraph :: !(TVar (Map Text (Set Text)))   -- File dependency graph (from trace)
  , fsDiagnostics :: !(TVar (Map Text [Diagnostic]))  -- Cached diagnostics
  }

data FileState = FileState
  { fsContent :: !Text
  , fsSymbols :: ![Symbol]
  , fsImports :: ![Text]       -- Files this file imports (from trace)
  , fsParsedAt :: !UTCTime
  }

-- | Initialize fallback state
initFallback :: IO FallbackState
initFallback = do
  files <- newTVarIO Map.empty
  graph <- newTVarIO Map.empty
  diags <- newTVarIO Map.empty
  pure FallbackState
    { fsFiles = files
    , fsGraph = graph
    , fsDiagnostics = diags
    }

-- -----------------------------------------------------------------------------
-- Main Server
-- -----------------------------------------------------------------------------

data LSPState = LSPState
  { lsConfig :: !LSPConfig
  , lsBackends :: !(TVar (Map Text BackendState))
  , lsFallback :: !FallbackState
  , lsInitialized :: !(TVar Bool)
  }

-- | Start the LSP server
startLSP :: LSPConfig -> IO ()
startLSP cfg = do
  backends <- newTVarIO Map.empty
  fallback <- initFallback
  initialized <- newTVarIO False
  
  let state = LSPState
        { lsConfig = cfg
        , lsBackends = backends
        , lsFallback = fallback
        , lsInitialized = initialized
        }
  
  -- Run main loop on stdio
  runStdioLoop state

-- | Main stdio loop
runStdioLoop :: LSPState -> IO ()
runStdioLoop state = forever $ do
  msg <- readMessage stdin
  case msg of
    Left err -> logError state $ "Parse error: " <> err
    Right (reqId, req) -> do
      resp <- handleRequest state req
      writeResponse stdout reqId resp

-- | Handle a single request
handleRequest :: LSPState -> Request -> IO Response
handleRequest state = \case
  Initialize params -> do
    atomically $ writeTVar (lsInitialized state) True
    pure $ InitializeResult $ Aeson.object
      [ "capabilities" .= capabilities ]
  
  Shutdown -> do
    -- Kill all backend processes
    backends <- readTVarIO (lsBackends state)
    forM_ (Map.elems backends) $ \bs ->
      case bsBackend bs of
        Just (RealLSP proc _ _ _) -> terminateProcess proc
        _ -> pure ()
    pure ShutdownResult
  
  TextDocumentCompletion uri pos -> do
    let lang = uriToLanguage uri
    handleWithFallback state lang
      (forwardCompletion uri pos)
      (fallbackCompletion (lsFallback state) uri pos)
  
  TextDocumentHover uri pos -> do
    let lang = uriToLanguage uri
    handleWithFallback state lang
      (forwardHover uri pos)
      (fallbackHover (lsFallback state) uri pos)
  
  TextDocumentDefinition uri pos -> do
    let lang = uriToLanguage uri
    handleWithFallback state lang
      (forwardDefinition uri pos)
      (fallbackDefinition (lsFallback state) uri pos)
  
  TextDocumentReferences uri pos -> do
    let lang = uriToLanguage uri
    handleWithFallback state lang
      (forwardReferences uri pos)
      (fallbackReferences (lsFallback state) uri pos)
  
  TextDocumentSymbol uri -> do
    let lang = uriToLanguage uri
    handleWithFallback state lang
      (forwardSymbols uri)
      (fallbackSymbols (lsFallback state) uri)
  
  TextDocumentDiagnostics uri -> 
    -- Diagnostics ALWAYS come from trace (real compiler)
    getDiagnosticsFromTrace (lsFallback state) uri
  
  TextDocumentDidOpen uri content -> do
    updateFileState (lsFallback state) uri content
    pure NullResult
  
  TextDocumentDidChange uri content -> do
    updateFileState (lsFallback state) uri content
    pure NullResult
  
  TextDocumentDidClose uri -> do
    -- Keep in cache for a while, don't eagerly remove
    pure NullResult
  
  WorkspaceSymbol query ->
    fallbackWorkspaceSymbols (lsFallback state) query
  
  UnknownRequest method _ -> do
    logInfo state $ "Unknown request: " <> method
    pure NullResult

-- | Handle request with fallback
handleWithFallback :: LSPState 
                   -> Text 
                   -> (Backend -> IO Response)  -- Real LSP handler
                   -> IO Response               -- Fallback handler
                   -> IO Response
handleWithFallback state lang realHandler fallbackHandler = do
  backend <- selectBackend (lsConfig state) (lsBackends state) lang
  case backend of
    Fallback -> fallbackHandler
    
    real@(RealLSP proc _ _ _) -> do
      -- Check if process is still alive
      exitCode <- getProcessExitCode proc
      case exitCode of
        Just _ -> do
          -- Process died, mark crashed and fallback
          updateBackendState (lsBackends state) lang $ \bs ->
            bs { bsStatus = BackendCrashed
               , bsCrashCount = bsCrashCount bs + 1
               , bsBackend = Nothing
               }
          fallbackHandler
        
        Nothing -> do
          -- Try real backend with timeout
          let timeoutUs = lcBackendTimeout (lsConfig state)
          result <- timeout timeoutUs (realHandler real)
          case result of
            Just resp -> do
              now <- getCurrentTime
              updateBackendState (lsBackends state) lang $ \bs ->
                bs { bsLastResponse = Just now
                   , bsSlowCount = 0
                   }
              pure resp
            
            Nothing -> do
              -- Too slow, use fallback
              updateBackendState (lsBackends state) lang $ \bs ->
                bs { bsStatus = BackendSlow
                   , bsSlowCount = bsSlowCount bs + 1
                   }
              fallbackHandler

-- -----------------------------------------------------------------------------
-- Forward to Real LSP
-- -----------------------------------------------------------------------------

forwardCompletion :: Text -> Position -> Backend -> IO Response
forwardCompletion uri pos (RealLSP _ stdin' stdout' _) = do
  -- TODO: Implement LSP JSON-RPC forwarding
  pure $ CompletionResult []
forwardCompletion _ _ Fallback = pure $ CompletionResult []

forwardHover :: Text -> Position -> Backend -> IO Response
forwardHover uri pos (RealLSP _ stdin' stdout' _) = do
  -- TODO: Implement LSP JSON-RPC forwarding
  pure $ HoverResult Nothing
forwardHover _ _ Fallback = pure $ HoverResult Nothing

forwardDefinition :: Text -> Position -> Backend -> IO Response
forwardDefinition uri pos (RealLSP _ stdin' stdout' _) = do
  -- TODO: Implement LSP JSON-RPC forwarding
  pure $ DefinitionResult []
forwardDefinition _ _ Fallback = pure $ DefinitionResult []

forwardReferences :: Text -> Position -> Backend -> IO Response
forwardReferences uri pos (RealLSP _ stdin' stdout' _) = do
  -- TODO: Implement LSP JSON-RPC forwarding
  pure $ ReferencesResult []
forwardReferences _ _ Fallback = pure $ ReferencesResult []

forwardSymbols :: Text -> Backend -> IO Response
forwardSymbols uri (RealLSP _ stdin' stdout' _) = do
  -- TODO: Implement LSP JSON-RPC forwarding
  pure $ SymbolResult []
forwardSymbols _ Fallback = pure $ SymbolResult []

-- -----------------------------------------------------------------------------
-- Fallback Handlers (tree-sitter based)
-- -----------------------------------------------------------------------------

fallbackCompletion :: FallbackState -> Text -> Position -> IO Response
fallbackCompletion fs uri pos = do
  files <- readTVarIO (fsFiles fs)
  graph <- readTVarIO (fsGraph fs)
  
  -- Get symbols from this file and its imports
  let thisFile = Map.lookup uri files
      importedUris = maybe [] fsImports thisFile
      importedFiles = mapMaybe (`Map.lookup` files) importedUris
      allSymbols = maybe [] fsSymbols thisFile 
                ++ concatMap fsSymbols importedFiles
  
  -- Convert symbols to completion items
  let items = map symbolToCompletion allSymbols
  pure $ CompletionResult items

symbolToCompletion :: Symbol -> CompletionItem
symbolToCompletion Symbol{..} = CompletionItem
  { ciLabel = symName
  , ciKind = Just symKind
  , ciDetail = symContainerName
  , ciDocumentation = Nothing
  , ciInsertText = Just symName
  }

fallbackHover :: FallbackState -> Text -> Position -> IO Response
fallbackHover fs uri pos = do
  files <- readTVarIO (fsFiles fs)
  case Map.lookup uri files of
    Nothing -> pure $ HoverResult Nothing
    Just fileState -> do
      -- Find symbol at position
      let syms = fsSymbols fileState
          atPos = filter (symbolContainsPosition pos) syms
      case atPos of
        [] -> pure $ HoverResult Nothing
        (sym:_) -> pure $ HoverResult $ Just $ 
          symName sym <> " :: " <> T.pack (show (symKind sym))

symbolContainsPosition :: Position -> Symbol -> Bool
symbolContainsPosition pos sym =
  let Range start end = locRange (symLocation sym)
  in start <= pos && pos <= end

fallbackDefinition :: FallbackState -> Text -> Position -> IO Response
fallbackDefinition fs uri pos = do
  files <- readTVarIO (fsFiles fs)
  graph <- readTVarIO (fsGraph fs)
  
  -- Find symbol at position
  case Map.lookup uri files of
    Nothing -> pure $ DefinitionResult []
    Just fileState -> do
      -- Get identifier at position (simplified - would need proper parsing)
      -- For now, find any symbol whose name might match
      let importedUris = fsImports fileState
          allFiles = Map.elems $ Map.restrictKeys files (Set.fromList (uri : importedUris))
          allSymbols = concatMap fsSymbols allFiles
      
      -- TODO: Get actual identifier at position
      -- For now, return empty
      pure $ DefinitionResult []

fallbackReferences :: FallbackState -> Text -> Position -> IO Response
fallbackReferences fs uri pos = do
  -- Would need to track symbol usages across files
  -- For now, return empty
  pure $ ReferencesResult []

fallbackSymbols :: FallbackState -> Text -> IO Response
fallbackSymbols fs uri = do
  files <- readTVarIO (fsFiles fs)
  case Map.lookup uri files of
    Nothing -> pure $ SymbolResult []
    Just fileState -> pure $ SymbolResult (fsSymbols fileState)

fallbackWorkspaceSymbols :: FallbackState -> Text -> IO Response
fallbackWorkspaceSymbols fs query = do
  files <- readTVarIO (fsFiles fs)
  let allSymbols = concatMap fsSymbols (Map.elems files)
      matching = filter (queryMatches query) allSymbols
  pure $ SymbolResult matching
  where
    queryMatches q sym = T.toLower q `T.isInfixOf` T.toLower (symName sym)

-- -----------------------------------------------------------------------------
-- Diagnostics from Trace
-- -----------------------------------------------------------------------------

getDiagnosticsFromTrace :: FallbackState -> Text -> IO Response
getDiagnosticsFromTrace fs uri = do
  diags <- readTVarIO (fsDiagnostics fs)
  pure $ DiagnosticsResult $ Map.findWithDefault [] uri diags

-- | Update diagnostics from compiler output
-- Called by trace watcher when compiler runs
updateDiagnosticsFromCompiler :: FallbackState -> Text -> Text -> IO ()
updateDiagnosticsFromCompiler fs uri compilerStderr = do
  let diags = parseCompilerDiagnostics uri compilerStderr
  atomically $ modifyTVar (fsDiagnostics fs) (Map.insert uri diags)

-- | Update file graph from trace
-- Called when we observe a compilation
updateGraphFromTrace :: FallbackState -> Trace.FullTrace -> IO ()
updateGraphFromTrace fs trace = do
  -- Extract which files were read (inputs) and written (outputs)
  let reads = [ Trace.faPath fa | fa <- Trace.ftFileAccesses trace
                                , Trace.faMode fa == Trace.ModeRead
                                , isSourceFile (Trace.faPath fa) ]
      writes = [ Trace.faPath fa | fa <- Trace.ftFileAccesses trace
                                 , Trace.faMode fa `elem` [Trace.ModeWrite, Trace.ModeCreate]
                                 , isOutputFile (Trace.faPath fa) ]
  
  -- For each output, record that it depends on all the reads
  forM_ writes $ \output -> do
    atomically $ modifyTVar (fsGraph fs) $ \g ->
      Map.insert output (Set.fromList reads) g
  
  -- Update import relationships in file state
  forM_ reads $ \readFile -> do
    files <- readTVarIO (fsFiles fs)
    case Map.lookup readFile files of
      Nothing -> pure ()  -- File not open in editor
      Just fileState -> do
        -- This file was read during compilation - update its imports
        let otherReads = filter (/= readFile) reads
        atomically $ modifyTVar (fsFiles fs) $ 
          Map.adjust (\s -> s { fsImports = nub (fsImports s ++ otherReads) }) readFile
  
  -- Update diagnostics from compiler stderr
  let stderr = Trace.ftStderr trace
  when (not $ T.null stderr) $ do
    -- Try to attribute diagnostics to specific files
    forM_ reads $ \file -> do
      let diags = parseCompilerDiagnostics file stderr
      when (not $ null diags) $
        atomically $ modifyTVar (fsDiagnostics fs) (Map.insert file diags)

isSourceFile :: Text -> Bool
isSourceFile path = any (`T.isSuffixOf` path)
  [".rs", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx"
  , ".hs", ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".zig"
  , ".nix", ".lean", ".ml", ".mli"]

isOutputFile :: Text -> Bool
isOutputFile path = any (`T.isSuffixOf` path)
  [".o", ".obj", ".a", ".so", ".dylib", ".dll", ".exe"
  , ".hi", ".chi", ".rlib", ".rmeta", ".d"]

nub :: Ord a => [a] -> [a]
nub = Set.toList . Set.fromList

-- | Parse compiler stderr into diagnostics
-- TODO: Add per-compiler parsers (rustc, clang, ghc, etc.)
parseCompilerDiagnostics :: Text -> Text -> [Diagnostic]
parseCompilerDiagnostics uri stderr =
  mapMaybe (parseLine uri) (T.lines stderr)
  where
    parseLine u line
      -- GCC/Clang format: file:line:col: error/warning: message
      | Just (file, rest) <- T.breakOn ":" line & breakOk
      , file == u || ("/" <> file) `T.isSuffixOf` u
      , Just (lineNo, rest') <- T.breakOn ":" (T.drop 1 rest) & breakOk
      , Just (colNo, rest'') <- T.breakOn ":" (T.drop 1 rest') & breakOk
      , Just (severity, msg) <- T.breakOn ":" (T.drop 1 rest'') & breakOk
      = Just Diagnostic
          { diagRange = Range 
              (Position (readInt lineNo - 1) (readInt colNo - 1))
              (Position (readInt lineNo - 1) (readInt colNo))
          , diagSeverity = parseSeverity severity
          , diagCode = Nothing
          , diagSource = Just "compiler"
          , diagMessage = T.strip (T.drop 1 msg)
          }
      | otherwise = Nothing
    
    breakOk (a, b) = if T.null b then Nothing else Just (a, b)
    readInt t = fromMaybe 0 $ readMaybe $ T.unpack t
    readMaybe s = case reads s of
      [(n, "")] -> Just n
      _ -> Nothing
    
    parseSeverity s
      | "error" `T.isInfixOf` s = Just Error
      | "warning" `T.isInfixOf` s = Just Warning
      | "note" `T.isInfixOf` s = Just Information
      | otherwise = Nothing

-- -----------------------------------------------------------------------------
-- File State Management
-- -----------------------------------------------------------------------------

updateFileState :: FallbackState -> Text -> Text -> IO ()
updateFileState fs uri content = do
  now <- getCurrentTime
  let symbols = parseSymbols uri content
  atomically $ modifyTVar (fsFiles fs) $ Map.insert uri FileState
    { fsContent = content
    , fsSymbols = symbols
    , fsImports = []  -- Will be updated from trace
    , fsParsedAt = now
    }

-- | Parse symbols from file content
-- TODO: Use actual tree-sitter
parseSymbols :: Text -> Text -> [Symbol]
parseSymbols uri content = 
  -- Simplified heuristic parser - would use tree-sitter
  mapMaybe (parseSymbolLine uri) (zip [0..] (T.lines content))

parseSymbolLine :: Text -> (Int, Text) -> Maybe Symbol
parseSymbolLine uri (lineNo, line)
  -- fn name(...) - Rust function
  | "fn " `T.isPrefixOf` stripped
  , Just name <- extractIdent (T.drop 3 stripped)
  = Just $ mkSymbol uri lineNo name SkFunction
  
  -- def name(...) - Python function
  | "def " `T.isPrefixOf` stripped
  , Just name <- extractIdent (T.drop 4 stripped)
  = Just $ mkSymbol uri lineNo name SkFunction
  
  -- struct Name - Rust struct
  | "struct " `T.isPrefixOf` stripped
  , Just name <- extractIdent (T.drop 7 stripped)
  = Just $ mkSymbol uri lineNo name SkStruct
  
  -- class Name - Python/TS class
  | "class " `T.isPrefixOf` stripped
  , Just name <- extractIdent (T.drop 6 stripped)
  = Just $ mkSymbol uri lineNo name SkClass
  
  -- const NAME - Rust/JS const
  | "const " `T.isPrefixOf` stripped
  , Just name <- extractIdent (T.drop 6 stripped)
  = Just $ mkSymbol uri lineNo name SkConstant
  
  -- let name - Rust/JS variable
  | "let " `T.isPrefixOf` stripped
  , Just name <- extractIdent (T.drop 4 stripped)
  = Just $ mkSymbol uri lineNo name SkVariable
  
  | otherwise = Nothing
  where
    stripped = T.stripStart line
    
    extractIdent t = 
      let ident = T.takeWhile isIdentChar t
      in if T.null ident then Nothing else Just ident
    
    isIdentChar c = c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9'
    
    mkSymbol u ln name kind = Symbol
      { symName = name
      , symKind = kind
      , symLocation = Location u (Range (Position ln 0) (Position ln (T.length line)))
      , symContainerName = Nothing
      }

-- -----------------------------------------------------------------------------
-- Utilities
-- -----------------------------------------------------------------------------

uriToLanguage :: Text -> Text
uriToLanguage uri = case takeExtension (T.unpack uri) of
  ".rs" -> "rust"
  ".c" -> "c"
  ".h" -> "c"
  ".cpp" -> "cpp"
  ".cc" -> "cpp"
  ".cxx" -> "cpp"
  ".hpp" -> "cpp"
  ".hxx" -> "cpp"
  ".hs" -> "haskell"
  ".py" -> "python"
  ".ts" -> "typescript"
  ".tsx" -> "typescript"
  ".js" -> "javascript"
  ".jsx" -> "javascript"
  ".go" -> "go"
  ".zig" -> "zig"
  ".nix" -> "nix"
  ".lean" -> "lean"
  _ -> "unknown"

capabilities :: Aeson.Value
capabilities = Aeson.object
  [ "textDocumentSync" .= Aeson.object
      [ "openClose" .= True
      , "change" .= (1 :: Int)  -- Full sync
      ]
  , "completionProvider" .= Aeson.object
      [ "triggerCharacters" .= [("." :: Text), "::", "->"]
      ]
  , "hoverProvider" .= True
  , "definitionProvider" .= True
  , "referencesProvider" .= True
  , "documentSymbolProvider" .= True
  , "workspaceSymbolProvider" .= True
  ]

-- -----------------------------------------------------------------------------
-- JSON-RPC (simplified)
-- -----------------------------------------------------------------------------

readMessage :: Handle -> IO (Either Text (Maybe Int, Request))
readMessage h = do
  -- Read headers
  headerLine <- hGetLine h
  let contentLength = parseContentLength (T.pack headerLine)
  
  -- Skip blank line
  _ <- hGetLine h
  
  -- Read body
  body <- B8.hGet h (fromMaybe 0 contentLength)
  
  case Aeson.eitherDecode (BL.fromStrict body) of
    Left err -> pure $ Left $ T.pack err
    Right obj -> pure $ Right (parseRequest obj)

parseContentLength :: Text -> Maybe Int
parseContentLength t
  | "Content-Length: " `T.isPrefixOf` t = readMaybe $ T.unpack $ T.drop 16 t
  | otherwise = Nothing
  where
    readMaybe s = case reads s of
      [(n, "")] -> Just n
      _ -> Nothing

parseRequest :: Aeson.Value -> (Maybe Int, Request)
parseRequest = \case
  Aeson.Object obj -> 
    let reqId = case AesonKM.lookup "id" obj of
          Just (Aeson.Number n) -> Just (round n)
          _ -> Nothing
        method = case AesonKM.lookup "method" obj of
          Just (Aeson.String m) -> m
          _ -> ""
        params = fromMaybe Aeson.Null $ AesonKM.lookup "params" obj
    in (reqId, methodToRequest method params)
  _ -> (Nothing, UnknownRequest "" Aeson.Null)

methodToRequest :: Text -> Aeson.Value -> Request
methodToRequest method params = case method of
  "initialize" -> Initialize params
  "shutdown" -> Shutdown
  "textDocument/completion" -> 
    case extractUriPos params of
      Just (uri, pos) -> TextDocumentCompletion uri pos
      Nothing -> UnknownRequest method params
  "textDocument/hover" ->
    case extractUriPos params of
      Just (uri, pos) -> TextDocumentHover uri pos
      Nothing -> UnknownRequest method params
  "textDocument/definition" ->
    case extractUriPos params of
      Just (uri, pos) -> TextDocumentDefinition uri pos
      Nothing -> UnknownRequest method params
  "textDocument/references" ->
    case extractUriPos params of
      Just (uri, pos) -> TextDocumentReferences uri pos
      Nothing -> UnknownRequest method params
  "textDocument/documentSymbol" ->
    case extractUri params of
      Just uri -> TextDocumentSymbol uri
      Nothing -> UnknownRequest method params
  "workspace/symbol" ->
    case params of
      Aeson.Object obj | Just (Aeson.String q) <- AesonKM.lookup "query" obj ->
        WorkspaceSymbol q
      _ -> UnknownRequest method params
  _ -> UnknownRequest method params

extractUri :: Aeson.Value -> Maybe Text
extractUri (Aeson.Object obj) = do
  Aeson.Object td <- AesonKM.lookup "textDocument" obj
  Aeson.String uri <- AesonKM.lookup "uri" td
  pure uri
extractUri _ = Nothing

extractUriPos :: Aeson.Value -> Maybe (Text, Position)
extractUriPos (Aeson.Object obj) = do
  uri <- extractUri (Aeson.Object obj)
  Aeson.Object posObj <- AesonKM.lookup "position" obj
  Aeson.Number line <- AesonKM.lookup "line" posObj
  Aeson.Number char <- AesonKM.lookup "character" posObj
  pure (uri, Position (round line) (round char))
extractUriPos _ = Nothing

writeResponse :: Handle -> Maybe Int -> Response -> IO ()
writeResponse h reqId resp = do
  let body = Aeson.encode $ responseToJson reqId resp
      len = BL.length body
      msg = "Content-Length: " <> show len <> "\r\n\r\n"
  hPutStr h msg
  BL.hPut h body
  hFlush h

responseToJson :: Maybe Int -> Response -> Aeson.Value
responseToJson reqId resp = Aeson.object $
  [ "jsonrpc" .= ("2.0" :: Text) ]
  ++ maybe [] (\i -> ["id" .= i]) reqId
  ++ case resp of
    InitializeResult caps -> ["result" .= caps]
    ShutdownResult -> ["result" .= Aeson.Null]
    CompletionResult items -> ["result" .= items]
    HoverResult mContent -> ["result" .= maybe Aeson.Null (\c -> Aeson.object ["contents" .= c]) mContent]
    DefinitionResult locs -> ["result" .= locs]
    ReferencesResult locs -> ["result" .= locs]
    SymbolResult syms -> ["result" .= syms]
    DiagnosticsResult diags -> ["result" .= diags]
    ErrorResponse code msg -> ["error" .= Aeson.object ["code" .= code, "message" .= msg]]
    NullResult -> ["result" .= Aeson.Null]

-- -----------------------------------------------------------------------------
-- Logging
-- -----------------------------------------------------------------------------

-- | Log to stderr (not stdout, which is used by LSP protocol)
logError :: LSPState -> Text -> IO ()
logError state msg = when (lcVerbose (lsConfig state)) $
  TIO.hPutStrLn stderr $ "[ERROR] " <> msg

logInfo :: LSPState -> Text -> IO ()
logInfo state msg = when (lcVerbose (lsConfig state)) $
  TIO.hPutStrLn stderr $ "[INFO] " <> msg

-- -----------------------------------------------------------------------------
-- Compile Commands (from shim metadata)
-- -----------------------------------------------------------------------------

-- | A compile command for compile_commands.json
data CompileCommand = CompileCommand
  { ccDirectory :: !Text      -- ^ Working directory
  , ccFile :: !Text           -- ^ Source file (absolute path)
  , ccArguments :: ![Text]    -- ^ Compile command as list
  , ccOutput :: !(Maybe Text) -- ^ Output file
  }
  deriving stock (Show, Eq, Generic)

instance ToJSON CompileCommand where
  toJSON CompileCommand{..} = Aeson.object $
    [ "directory" .= ccDirectory
    , "file" .= ccFile
    , "arguments" .= ccArguments
    ] ++ maybe [] (\o -> ["output" .= o]) ccOutput

-- | Generate compile_commands.json from shim build output
-- Reads the final executable and extracts all compile info from metadata
generateCompileCommands :: FilePath -> FilePath -> IO [CompileCommand]
generateCompileCommands workDir targetPath = do
  meta <- Shim.readBuildMetadata targetPath
  case meta of
    Nothing -> pure []
    Just bm -> do
      -- Each source file becomes a compile command
      let sources = Shim.bmSources bm
          includes = Shim.bmIncludes bm
          defines = Shim.bmDefines bm
          flags = Shim.bmFlags bm
      
      pure [ CompileCommand
               { ccDirectory = T.pack workDir
               , ccFile = src
               , ccArguments = buildArgs src includes defines flags
               , ccOutput = Just $ T.replace (T.pack ".c") (T.pack ".o") $
                            T.replace (T.pack ".cpp") (T.pack ".o") $
                            T.replace (T.pack ".cc") (T.pack ".o") src
               }
           | src <- sources
           ]
  where
    buildArgs src includes defines flags =
      ["clang++"]  -- Assume clang++, could be extracted from shim
      ++ ["-c", src]
      ++ map ("-I" <>) includes
      ++ map ("-D" <>) defines
      ++ flags

-- | Write compile_commands.json to a file
writeCompileCommands :: FilePath -> [CompileCommand] -> IO ()
writeCompileCommands path cmds = do
  let json = Aeson.encode cmds
  BL.writeFile path json

-- | Update fallback state from shim build output
updateFromShimBuild :: FallbackState -> FilePath -> IO ()
updateFromShimBuild fs targetPath = do
  meta <- Shim.readBuildMetadata targetPath
  case meta of
    Nothing -> pure ()
    Just bm -> do
      -- Update file graph: each source depends on includes
      let sources = Shim.bmSources bm
          includes = Shim.bmIncludes bm
      
      -- For each source, record its "imports" (the includes it uses)
      forM_ sources $ \src -> do
        now <- getCurrentTime
        atomically $ modifyTVar (fsFiles fs) $ \files ->
          case Map.lookup src files of
            Just existing -> 
              Map.insert src existing { fsImports = map T.pack (filter isHeader (map T.unpack includes)) } files
            Nothing -> files
      
      -- Update dependency graph
      atomically $ modifyTVar (fsGraph fs) $ \g ->
        Map.insert (Shim.bmTarget bm) (Set.fromList sources) g
  where
    isHeader path = any (`isSuffixOf` path) [".h", ".hpp", ".hxx", ".H"]
    isSuffixOf suffix s = suffix == drop (length s - length suffix) s
