{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Armitage.DICE
Description : Incremental computation graph for builds

DICE (Dynamic Incremental Computation Engine) - the good part of Buck2.

Key insight: everything is content-addressed.
  ActionKey = hash(inputs + command)
  
If inputs unchanged → outputs unchanged → skip execution.

This module handles:
  1. Graph construction from Dhall targets
  2. Flake reference resolution (nix build --print-out-paths)
  3. Topological execution with CAS caching
  4. Coeffect tracking per action

The graph can mix:
  - Local targets (we build)
  - Flake refs (nix builds, we consume)
  - CA fetches (curl + hash check)
-}
module Armitage.DICE
  ( -- * Keys
    ActionKey (..)
  , actionKey
  
    -- * Actions
  , Action (..)
  , ActionCategory (..)
  , Input (..)
  , ResolvedInput (..)
  
    -- * Graph
  , ActionGraph (..)
  , buildGraph
  , topoSort
  
    -- * Resolution
  , resolveFlake
  , resolveFlakeBatch
  , ResolvedFlake (..)
  
    -- * Execution
  , ExecutionResult (..)
  , ExecutionMode (..)
  , WitnessConfig (..)
  , executeGraph
  , executeGraphWitnessed
  , executeGraphRemote
  , executeAction
  , executeActionWitnessed
  , executeActionRemote
  
    -- * Analysis
  , analyze
  , AnalysisResult (..)
  ) where

import Control.Concurrent.Async (mapConcurrently)
import Control.Monad (forM, foldM, unless, when)
import Crypto.Hash (SHA256 (..), hashWith)
import Data.ByteArray.Encoding qualified as BA
import Data.ByteString ()
import qualified Data.ByteString as BS

import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.IO as TIO
import Data.Time.Clock (getCurrentTime)
import GHC.Generics (Generic)
import System.Directory (doesFileExist, doesPathExist, createDirectoryIfMissing)
import System.Exit (ExitCode (..))
import System.IO (hFlush, stdout)
import System.Process (readProcessWithExitCode)

import qualified Armitage.Dhall as Dhall
import qualified Armitage.Builder as Builder
import qualified Armitage.CAS as CAS
import qualified Armitage.Proto as Proto
import qualified Armitage.RE as RE

import Data.Aeson (FromJSON)
import qualified Data.Aeson as Aeson
import Data.Maybe (catMaybes, fromMaybe)
import System.Environment (getEnvironment)
import System.FilePath ((</>), takeFileName)
import System.Process (readCreateProcessWithExitCode, proc, CreateProcess(..))

-- -----------------------------------------------------------------------------
-- Action Keys
-- -----------------------------------------------------------------------------

-- | Content-addressed action key
newtype ActionKey = ActionKey { unActionKey :: Text }
  deriving stock (Show, Eq, Ord, Generic)

-- | Compute action key from action content (canonical Dhall serialization)
actionKey :: Action -> ActionKey
actionKey action = 
  let content = actionToDhall action  -- Canonical Dhall representation
      hash = hashWith SHA256 (TE.encodeUtf8 content)
  in ActionKey $ TE.decodeUtf8 $ BA.convertToBase BA.Base16 hash

-- -----------------------------------------------------------------------------
-- Actions
-- -----------------------------------------------------------------------------

-- | What kind of action
data ActionCategory
  = CxxCompile        -- .cpp -> .o
  | CxxLink           -- .o -> binary/library
  | CudaCompile       -- .cu -> .o
  | CudaLink          -- .o -> binary with device code
  | Archive           -- .o -> .a
  | Shell             -- run a shell script
  | Custom Text       -- escape hatch
  deriving stock (Show, Eq, Generic)

-- | An input to an action (before resolution)
data Input
  = Input_Local Text              -- local target ":foo"
  | Input_Flake Text              -- "nixpkgs#openssl"
  | Input_Fetch Text Text         -- url, expected hash
  | Input_Source FilePath         -- source file
  deriving stock (Show, Eq, Generic)

-- | A resolved input (after analysis)
data ResolvedInput
  = Resolved_Action ActionKey     -- depends on another action's output
  | Resolved_Store Text           -- nix store path
  | Resolved_File FilePath        -- source file (hashed)
  deriving stock (Show, Eq, Ord, Generic)

-- | An action in the build graph
data Action = Action
  { aCategory    :: ActionCategory
  , aIdentifier  :: Text                    -- "//src:libfoo"
  , aInputs      :: [ResolvedInput]         -- resolved dependencies
  , aToolchain   :: Dhall.Toolchain
  , aCommand     :: [Text]                  -- actual command to run
  , aEnv         :: Map Text Text           -- environment
  , aOutputs     :: [Text]                  -- output names
  , aCoeffects   :: [Dhall.Resource]        -- what this action requires
  }
  deriving stock (Show, Eq, Generic)

-- -----------------------------------------------------------------------------
-- Dhall Serialization (canonical, for content-addressing)
-- -----------------------------------------------------------------------------

-- | Serialize ActionCategory to canonical Dhall
actionCategoryToDhall :: ActionCategory -> Text
actionCategoryToDhall = \case
  CxxCompile   -> "<CxxCompile>"
  CxxLink      -> "<CxxLink>"
  CudaCompile  -> "<CudaCompile>"
  CudaLink     -> "<CudaLink>"
  Archive      -> "<Archive>"
  Shell        -> "<Shell>"
  Custom t     -> "<Custom>.\"" <> escapeTextDice t <> "\""

-- | Serialize ResolvedInput to canonical Dhall
resolvedInputToDhall :: ResolvedInput -> Text
resolvedInputToDhall = \case
  Resolved_Action (ActionKey k) -> "<Resolved_Action>.\"" <> escapeTextDice k <> "\""
  Resolved_Store path           -> "<Resolved_Store>.\"" <> escapeTextDice path <> "\""
  Resolved_File path            -> "<Resolved_File>.\"" <> escapeTextDice (T.pack path) <> "\""

-- | Serialize Action to canonical Dhall (deterministic field order)
actionToDhall :: Action -> Text
actionToDhall Action{..} = T.unlines
  [ "{ aCategory = " <> actionCategoryToDhall aCategory
  , ", aCommand = " <> listToDhall aCommand
  , ", aCoeffects = [" <> T.intercalate ", " (map Dhall.renderResource aCoeffects) <> "]"
  , ", aEnv = " <> mapToDhall aEnv
  , ", aIdentifier = \"" <> escapeTextDice aIdentifier <> "\""
  , ", aInputs = [" <> T.intercalate ", " (map resolvedInputToDhall aInputs) <> "]"
  , ", aOutputs = " <> listToDhall aOutputs
  , ", aToolchain = " <> Dhall.renderToolchain aToolchain
  , "}"
  ]

-- | Escape text for Dhall string literal (local to DICE)
escapeTextDice :: Text -> Text
escapeTextDice = T.concatMap escapeChar
  where
    escapeChar c = case c of
      '"'  -> "\\\""
      '\\' -> "\\\\"
      '\n' -> "\\n"
      '\t' -> "\\t"
      '\r' -> "\\r"
      _    -> T.singleton c

-- | Render list to Dhall
listToDhall :: [Text] -> Text
listToDhall [] = "[] : List Text"
listToDhall xs = "[" <> T.intercalate ", " (map (\x -> "\"" <> escapeTextDice x <> "\"") xs) <> "]"

-- | Render map to Dhall (sorted keys for determinism)
mapToDhall :: Map Text Text -> Text
mapToDhall m
  | Map.null m = "toMap {=}"
  | otherwise  = "toMap {" <> T.intercalate ", " entries <> "}"
  where
    entries = [ "`" <> k <> "` = \"" <> escapeTextDice v <> "\""
              | (k, v) <- Map.toAscList m  -- sorted for determinism
              ]

-- -----------------------------------------------------------------------------
-- Action Graph
-- -----------------------------------------------------------------------------

-- | The full build graph
data ActionGraph = ActionGraph
  { agActions  :: Map ActionKey Action
  , agRoots    :: [ActionKey]               -- what we want to build
  , agDeps     :: Map ActionKey [ActionKey] -- adjacency list
  }
  deriving stock (Show, Generic)

-- | Topologically sort actions (deps before dependents)
topoSort :: ActionGraph -> [ActionKey]
topoSort ActionGraph {..} = reverse $ go Set.empty [] (Map.keys agActions)
  where
    go :: Set ActionKey -> [ActionKey] -> [ActionKey] -> [ActionKey]
    go _ sorted [] = sorted
    go visited sorted (k:ks)
      | k `Set.member` visited = go visited sorted ks
      | otherwise = 
          let deps = Map.findWithDefault [] k agDeps
              (visited', sorted') = foldl visitDep (Set.insert k visited, sorted) deps
          in go visited' (k : sorted') ks
    
    visitDep (v, s) dep
      | dep `Set.member` v = (v, s)
      | otherwise = 
          let deps = Map.findWithDefault [] dep agDeps
              (v', s') = foldl visitDep (Set.insert dep v, s) deps
          in (v', dep : s')

-- -----------------------------------------------------------------------------
-- Flake Resolution
-- -----------------------------------------------------------------------------

-- | Resolved flake reference
data ResolvedFlake = ResolvedFlake
  { rfRef       :: Text         -- original ref "nixpkgs#openssl"
  , rfStorePath :: Text         -- /nix/store/abc123-openssl-3.0.12
  , rfOutputs   :: Map Text Text -- out -> /nix/store/..., dev -> /nix/store/...
  }
  deriving stock (Show, Eq, Generic)

-- | Resolve a single flake reference
-- TODO: This should use wrappers that know the deps, with strace as verification.
-- Currently falls back to nix build (slow).
resolveFlake :: Text -> IO (Either Text ResolvedFlake)
resolveFlake ref = do
  -- Resolve default output
  (exitCode1, stdout1, _stderr1) <- readProcessWithExitCode 
    "nix" 
    ["build", "--no-link", "--print-out-paths", T.unpack ref] 
    ""
  
  -- Try to resolve dev output (may not exist)
  let devRef = ref <> ".dev"
  (exitCode2, stdout2, _) <- readProcessWithExitCode
    "nix"
    ["build", "--no-link", "--print-out-paths", T.unpack devRef]
    ""
  
  case exitCode1 of
    ExitFailure code -> 
      pure $ Left $ "Failed to resolve " <> ref <> " (exit " <> T.pack (show code) <> ")"
    ExitSuccess -> do
      let outPath = T.strip $ T.pack stdout1
          devPath = case exitCode2 of
            ExitSuccess -> Just $ T.strip $ T.pack stdout2
            _ -> Nothing
          outputs = Map.fromList $ 
            [("out", outPath)] ++ 
            maybe [] (\p -> [("dev", p)]) devPath
      pure $ Right ResolvedFlake
          { rfRef = ref
          , rfStorePath = outPath
          , rfOutputs = outputs
          }

-- | Resolve multiple flake references in parallel
resolveFlakeBatch :: [Text] -> IO (Map Text (Either Text ResolvedFlake))
resolveFlakeBatch refs = do
  results <- mapConcurrently resolveFlake refs
  pure $ Map.fromList $ zip refs results

-- -----------------------------------------------------------------------------
-- Analysis (Target -> ActionGraph)
-- -----------------------------------------------------------------------------

-- | Result of analyzing a target
data AnalysisResult = AnalysisResult
  { arGraph       :: ActionGraph
  , arFlakes      :: Map Text ResolvedFlake   -- resolved flake refs
  , arErrors      :: [Text]                   -- any resolution failures
  }
  deriving stock (Show, Generic)

-- | Analyze a target and its dependencies into an action graph
analyze :: Dhall.Target -> IO AnalysisResult
analyze target = do
  -- 1. Collect all flake refs from target and deps
  let flakeRefs = collectFlakeRefs target
  
  -- 2. Resolve them in parallel
  resolved <- resolveFlakeBatch (Set.toList flakeRefs)
  
  -- 3. Check for errors
  let errors = [ e | Left e <- Map.elems resolved ]
      flakes = Map.fromList [ (r, f) | (r, Right f) <- Map.toList resolved ]
  
  -- 4. Build action graph
  let graph = buildGraph target flakes
  
  pure AnalysisResult
    { arGraph = graph
    , arFlakes = flakes
    , arErrors = errors
    }

-- | Collect all flake references from a target
collectFlakeRefs :: Dhall.Target -> Set Text
collectFlakeRefs target = Set.fromList
  [ ref | Dhall.Dep_Flake ref <- Dhall.deps target ]

-- | Build action graph from resolved deps
buildGraph :: Dhall.Target -> Map Text ResolvedFlake -> ActionGraph
buildGraph target flakes = 
  let -- Create main action for this target
      mainAction = targetToAction target flakes
      mainKey = actionKey mainAction
      
      -- For now, single action (would recursively process deps)
      actions = Map.singleton mainKey mainAction
      
  in ActionGraph
      { agActions = actions
      , agRoots = [mainKey]
      , agDeps = Map.singleton mainKey []  -- deps already resolved to store paths
      }

-- | Convert a Dhall target to an Action
targetToAction :: Dhall.Target -> Map Text ResolvedFlake -> Action
targetToAction Dhall.Target {..} flakes = Action
  { aCategory = category
  , aIdentifier = targetName
  , aInputs = resolvedInputs
  , aToolchain = toolchain
  , aCommand = buildCommand
  , aEnv = Map.empty
  , aOutputs = [targetName]
  , aCoeffects = requires
  }
  where
    resolvedInputs = concatMap (resolveDep flakes) deps ++ sourceInputs
    
    sourceInputs = case srcs of
      Dhall.Src_Files fs -> map (Resolved_File . T.unpack) fs
      _ -> []
    
    resolveDep :: Map Text ResolvedFlake -> Dhall.Dep -> [ResolvedInput]
    resolveDep fs = \case
      Dhall.Dep_Local t -> [Resolved_Action (ActionKey t)]  -- would need recursive resolution
      Dhall.Dep_Flake ref -> 
        case Map.lookup ref fs of
          Just rf -> 
            -- Include both out and dev outputs if available
            let outPaths = Map.elems (rfOutputs rf)
            in map Resolved_Store outPaths
          Nothing -> []  -- error case, should have been caught
      Dhall.Dep_PkgConfig _ -> []  -- handled via pkg-config at compile time
      Dhall.Dep_External {} -> []  -- TODO: CA fetch
    
    -- Detect shell scripts vs C++ builds
    (category, buildCommand) = case srcs of
      Dhall.Src_Files [f] | ".sh" `T.isSuffixOf` f -> 
        (Shell, ["bash", f])
      _ -> 
        (CxxLink, cxxCommand)
    
    cxxCommand = 
      [ compilerCmd
      , "-o", targetName
      ] ++ flagsToArgs (Dhall.cflags toolchain)
        ++ includeArgs
        ++ sourceArgs
        ++ libArgs
        ++ linkLibs
    
    -- Compiler command - resolve from PATH or toolchain config
    -- The actual path should come from environment or be resolved by the executor
    compilerCmd = case Dhall.compiler toolchain of
      Dhall.Compiler_Clang {} -> "clang++"  -- Executor must have clang in PATH
      Dhall.Compiler_NVClang {} -> "clang++"
      Dhall.Compiler_GCC {} -> "g++"
      Dhall.Compiler_NVCC {} -> "nvcc"
      _ -> "cc"
    
    -- Extract library names from flake deps (heuristic: use package name)
    -- Some packages have different lib names than package names
    linkLibs = concatMap extractLibFlag deps
    extractLibFlag = \case
      Dhall.Dep_Flake ref -> 
        -- Extract lib name from "nixpkgs#foo" -> "foo"
        let name = T.takeWhileEnd (/= '#') ref
            libName = mapLibName name
        -- Skip header-only libs
        in if name `elem` ["nlohmann_json"]
           then []
           else ["-l" <> libName]
      _ -> []
    
    -- Map package names to actual library names
    mapLibName n = case n of
      "zlib" -> "z"
      "openssl" -> "ssl"
      "libpng" -> "png"
      "libjpeg" -> "jpeg"
      "sqlite" -> "sqlite3"
      "curl" -> "curl"
      _ -> n  -- default: same as package name
    
    sourceArgs = case srcs of
      Dhall.Src_Files fs -> fs
      _ -> []
    
    -- -I and -L from resolved flakes
    includeArgs = concatMap mkInclude resolvedInputs
    mkInclude (Resolved_Store p) = ["-I" <> p <> "/include"]
    mkInclude _ = []
    
    libArgs = concatMap mkLib resolvedInputs
    mkLib (Resolved_Store p) = ["-L" <> p <> "/lib"]
    mkLib _ = []

-- | Convert CFlags to command line args
flagsToArgs :: [Dhall.CFlag] -> [Text]
flagsToArgs = map Dhall.renderCFlag

-- -----------------------------------------------------------------------------
-- Execution
-- -----------------------------------------------------------------------------

-- | Execution mode - local or remote
data ExecutionMode
  = ExecLocal                    -- Local execution (readProcessWithExitCode)
  | ExecRemote RE.REConfig       -- Remote execution via NativeLink
  deriving stock (Show, Generic)

-- | Witness proxy configuration
data WitnessConfig = WitnessConfig
  { wcProxyHost :: String        -- Proxy host (e.g., "localhost")
  , wcProxyPort :: Int           -- Proxy port (e.g., 8080)
  , wcCertFile :: FilePath       -- CA cert for TLS MITM
  , wcLogDir :: FilePath         -- Where proxy writes attestations
  }
  deriving stock (Show, Generic)

-- | Result of executing the graph
data ExecutionResult = ExecutionResult
  { erOutputs   :: Map ActionKey [Text]     -- action -> output paths
  , erCacheHits :: Int
  , erExecuted  :: Int
  , erFailed    :: [(ActionKey, Text)]      -- failures
  , erProofs    :: Map ActionKey Builder.DischargeProof
  }
  deriving stock (Show, Generic)

-- | Execute the action graph (local, no witness)
executeGraph :: ActionGraph -> IO ExecutionResult
executeGraph = executeGraphWith ExecLocal Nothing

-- | Execute the action graph (local, with witness proxy)
executeGraphWitnessed :: WitnessConfig -> ActionGraph -> IO ExecutionResult
executeGraphWitnessed wc = executeGraphWith ExecLocal (Just wc)

-- | Execute the action graph (remote via NativeLink)
executeGraphRemote :: RE.REConfig -> ActionGraph -> IO ExecutionResult
executeGraphRemote config = executeGraphWith (ExecRemote config) Nothing

-- | Execute with specified mode
executeGraphWith :: ExecutionMode -> Maybe WitnessConfig -> ActionGraph -> IO ExecutionResult
executeGraphWith mode mWitness graph = do
  let sorted = topoSort graph
  
  case mode of
    ExecLocal -> do
      (outputs, hits, executed, failed, proofs) <- 
        foldM (executeOneLocal mWitness graph) (Map.empty, 0, 0, [], Map.empty) sorted
      pure ExecutionResult
        { erOutputs = outputs
        , erCacheHits = hits
        , erExecuted = executed
        , erFailed = failed
        , erProofs = proofs
        }
    
    ExecRemote config -> RE.withREClient config $ \client -> do
      (outputs, hits, executed, failed, proofs) <- 
        foldM (executeOneRemote client graph) (Map.empty, 0, 0, [], Map.empty) sorted
      pure ExecutionResult
        { erOutputs = outputs
        , erCacheHits = hits
        , erExecuted = executed
        , erFailed = failed
        , erProofs = proofs
        }

executeOneLocal 
  :: Maybe WitnessConfig
  -> ActionGraph 
  -> (Map ActionKey [Text], Int, Int, [(ActionKey, Text)], Map ActionKey Builder.DischargeProof)
  -> ActionKey
  -> IO (Map ActionKey [Text], Int, Int, [(ActionKey, Text)], Map ActionKey Builder.DischargeProof)
executeOneLocal mWitness graph (outputs, hits, executed, failed, proofs) key = do
  let action = agActions graph Map.! key
  
  -- Check CAS for cached output
  cached <- checkCAS key
  case cached of
    Just paths -> 
      -- Cache hit
      pure (Map.insert key paths outputs, hits + 1, executed, failed, proofs)
    
    Nothing -> do
      -- Cache miss - execute (with or without witness)
      result <- case mWitness of
        Just wc -> executeActionWitnessed wc action outputs
        Nothing -> executeAction action outputs
      case result of
        Left err -> 
          pure (outputs, hits, executed, (key, err) : failed, proofs)
        Right (paths, proof) -> do
          -- Store in CAS
          storeCAS key paths
          pure (Map.insert key paths outputs, hits, executed + 1, failed, Map.insert key proof proofs)

executeOneRemote 
  :: RE.REClient
  -> ActionGraph 
  -> (Map ActionKey [Text], Int, Int, [(ActionKey, Text)], Map ActionKey Builder.DischargeProof)
  -> ActionKey
  -> IO (Map ActionKey [Text], Int, Int, [(ActionKey, Text)], Map ActionKey Builder.DischargeProof)
executeOneRemote client graph (outputs, hits, executed, failed, proofs) key = do
  let action = agActions graph Map.! key
  
  -- Check action cache first (use canonical Dhall serialization for digest)
  let actionDigest = CAS.digestFromBytes $ TE.encodeUtf8 $ actionToDhall action
  cached <- RE.getActionResult client actionDigest
  case cached of
    Just result -> do
      -- Cache hit - extract output paths from result
      let paths = map RE.ofPath (RE.arOutputFiles result)
      pure (Map.insert key paths outputs, hits + 1, executed, failed, proofs)
    
    Nothing -> do
      -- Cache miss - execute remotely
      result <- executeActionRemote client action outputs
      case result of
        Left err -> 
          pure (outputs, hits, executed, (key, err) : failed, proofs)
        Right (paths, proof) -> 
          pure (Map.insert key paths outputs, hits, executed + 1, failed, Map.insert key proof proofs)

-- | Execute a single action
executeAction :: Action -> Map ActionKey [Text] -> IO (Either Text ([Text], Builder.DischargeProof))
executeAction action@Action {..} _depOutputs = do
  startTime <- getCurrentTime
  
  -- Check coeffects
  coeffectResult <- checkCoeffects aCoeffects
  case coeffectResult of
    Left missing -> pure $ Left $ "Missing coeffect: " <> T.pack (show missing)
    Right () -> do
      -- Build the full command
      let cmd = map T.unpack aCommand
      
      case cmd of
        [] -> pure $ Left "Empty command"
        (exe:args) -> do
          -- Print what we're running (full command for debugging)
          TIO.putStrLn $ "  $ " <> T.pack exe <> " \\"
          mapM_ (\a -> TIO.putStrLn $ "      " <> T.pack a) args
          hFlush stdout
          
          -- Execute
          (exitCode, _stdout, stderr) <- readProcessWithExitCode exe args ""
          endTime <- getCurrentTime
          
          case exitCode of
            ExitSuccess -> do
              -- Output path is just the target name for now
              -- Real impl would put in store
              let outputPaths = aOutputs
              
              -- Hash outputs for attestation
              outputHashes <- forM outputPaths $ \out -> do
                let outPath = T.unpack out
                exists <- doesFileExist outPath
                if exists
                  then do
                    content <- BS.readFile outPath
                    let hash = TE.decodeUtf8 $ BA.convertToBase BA.Base16 $ hashWith SHA256 content
                    pure (out, hash)
                  else pure (out, "missing")
              
              -- Create discharge proof
              let proof = Builder.DischargeProof
                    { Builder.dpCoeffects = map resourceToCoeffect aCoeffects
                    , Builder.dpNetworkAccess = []
                    , Builder.dpFilesystemAccess = []
                    , Builder.dpAuthUsage = []
                    , Builder.dpBuildId = unActionKey $ actionKey action
                    , Builder.dpDerivationHash = unActionKey $ actionKey action
                    , Builder.dpOutputHashes = outputHashes
                    , Builder.dpStartTime = startTime
                    , Builder.dpEndTime = endTime
                    }
              pure $ Right (outputPaths, proof)
            
            ExitFailure code -> do
              -- Print stderr on failure
              unless (null stderr) $
                TIO.putStrLn $ T.pack stderr
              pure $ Left $ "Exit code " <> T.pack (show code)

-- | Network attestation from witness proxy log
data WitnessAttestation = WitnessAttestation
  { waUrl :: Text
  , waHost :: Text
  , waSha256 :: Maybe Text
  , waSize :: Int
  , waTimestamp :: Text
  , waMethod :: Text
  , waCached :: Bool
  } deriving (Show, Generic)

instance FromJSON WitnessAttestation where
  parseJSON = Aeson.withObject "WitnessAttestation" $ \v -> WitnessAttestation
    <$> v Aeson..: "url"
    <*> v Aeson..: "host"
    <*> v Aeson..:? "sha256"
    <*> v Aeson..: "size"
    <*> v Aeson..: "timestamp"
    <*> v Aeson..: "method"
    <*> v Aeson..: "cached"

-- | Execute action with witness proxy (collects network attestations)
executeActionWitnessed :: WitnessConfig -> Action -> Map ActionKey [Text] -> IO (Either Text ([Text], Builder.DischargeProof))
executeActionWitnessed wc action@Action {..} _depOutputs = do
  startTime <- getCurrentTime
  
  -- Check coeffects (witnessed mode can satisfy Network)
  coeffectResult <- checkCoeffectsWitnessed wc aCoeffects
  case coeffectResult of
    Left missing -> pure $ Left $ "Missing coeffect: " <> T.pack (show missing)
    Right () -> do
      let cmd = map T.unpack aCommand
      
      case cmd of
        [] -> pure $ Left "Empty command"
        (exe:args) -> do
          -- Ensure log dir exists and clear attestation log
          let attestationLog = wcLogDir wc </> "fetches.jsonl"
          createDirectoryIfMissing True (wcLogDir wc)
          writeFile attestationLog ""
          
          -- Set up proxy environment
          let proxyUrl = "http://" <> wcProxyHost wc <> ":" <> show (wcProxyPort wc)
              proxyEnv = [ ("HTTP_PROXY", proxyUrl)
                        , ("HTTPS_PROXY", proxyUrl)
                        , ("http_proxy", proxyUrl)
                        , ("https_proxy", proxyUrl)
                        , ("SSL_CERT_FILE", wcCertFile wc)
                        , ("NIX_SSL_CERT_FILE", wcCertFile wc)
                        ]
          
          -- Print what we're running
          TIO.putStrLn $ "  $ " <> T.pack exe <> " \\"
          mapM_ (\a -> TIO.putStrLn $ "      " <> T.pack a) args
          TIO.putStrLn $ "  [via witness proxy " <> T.pack proxyUrl <> "]"
          hFlush stdout
          
          -- Get current env and merge with proxy env
          currentEnv <- getEnvironment
          let fullEnv = proxyEnv ++ currentEnv
          
          -- Execute with proxy env
          let procSpec = (System.Process.proc exe args) { env = Just fullEnv }
          (exitCode, _stdout, stderr) <- readCreateProcessWithExitCode procSpec ""
          endTime <- getCurrentTime
          
          case exitCode of
            ExitSuccess -> do
              let outputPaths = aOutputs
              
              -- Hash outputs
              outputHashes <- forM outputPaths $ \out -> do
                let outPath = T.unpack out
                exists <- doesFileExist outPath
                if exists
                  then do
                    content <- BS.readFile outPath
                    let hash = TE.decodeUtf8 $ BA.convertToBase BA.Base16 $ hashWith SHA256 content
                    pure (out, hash)
                  else pure (out, "missing")
              
              -- Read attestations from proxy log
              networkAccess <- readAttestations attestationLog
              
              -- Check for coeffect violations
              let declaredPure = null aCoeffects || all (== Dhall.Resource_Pure) aCoeffects
                  hasNetwork = not (null networkAccess)
              when (declaredPure && hasNetwork) $
                TIO.putStrLn $ "  ⚠ COEFFECT VIOLATION: declared pure but made " 
                             <> T.pack (show (length networkAccess)) <> " network request(s)"
              
              let proof = Builder.DischargeProof
                    { Builder.dpCoeffects = map resourceToCoeffect aCoeffects
                    , Builder.dpNetworkAccess = networkAccess
                    , Builder.dpFilesystemAccess = []
                    , Builder.dpAuthUsage = []
                    , Builder.dpBuildId = unActionKey $ actionKey action
                    , Builder.dpDerivationHash = unActionKey $ actionKey action
                    , Builder.dpOutputHashes = outputHashes
                    , Builder.dpStartTime = startTime
                    , Builder.dpEndTime = endTime
                    }
              pure $ Right (outputPaths, proof)
            
            ExitFailure code -> do
              unless (null stderr) $ TIO.putStrLn $ T.pack stderr
              pure $ Left $ "Exit code " <> T.pack (show code)

-- | Read attestations from witness proxy log
readAttestations :: FilePath -> IO [Builder.NetworkAccess]
readAttestations logPath = do
  exists <- doesFileExist logPath
  if not exists
    then pure []
    else do
      content <- TIO.readFile logPath
      let lines' = filter (not . T.null) $ T.lines content
      attestations <- forM lines' $ \line -> do
        case Aeson.eitherDecodeStrict (TE.encodeUtf8 line) of
          Left _ -> pure Nothing
          Right (wa :: WitnessAttestation) -> do
            now <- getCurrentTime
            pure $ Just Builder.NetworkAccess
              { Builder.naUrl = waUrl wa
              , Builder.naMethod = waMethod wa
              , Builder.naContentHash = fromMaybe "" (waSha256 wa)
              , Builder.naTimestamp = now  -- TODO: parse waTimestamp
              }
      pure $ catMaybes attestations

-- | Execute a single action remotely via NativeLink
executeActionRemote :: RE.REClient -> Action -> Map ActionKey [Text] -> IO (Either Text ([Text], Builder.DischargeProof))
executeActionRemote client action@Action {..} _depOutputs = do
  startTime <- getCurrentTime
  
  -- 1. Build Command proto
  let command = RE.Command
        { RE.cmdArguments = aCommand
        , RE.cmdEnvironmentVariables = Map.toList aEnv
        , RE.cmdOutputFiles = aOutputs
        , RE.cmdOutputDirectories = []
        , RE.cmdWorkingDirectory = ""
        , RE.cmdOutputPaths = aOutputs
        }
  
  -- 2. Serialize and upload Command to CAS (proper protobuf encoding)
  let protoCommand = Proto.ProtoCommand
        { Proto.pcArguments = RE.cmdArguments command
        , Proto.pcEnvironmentVariables = 
            [ Proto.ProtoEnvironmentVariable k v 
            | (k, v) <- RE.cmdEnvironmentVariables command 
            ]
        , Proto.pcOutputFiles = RE.cmdOutputFiles command
        , Proto.pcOutputDirectories = RE.cmdOutputDirectories command
        , Proto.pcWorkingDirectory = RE.cmdWorkingDirectory command
        , Proto.pcOutputPaths = RE.cmdOutputPaths command
        }
      commandBytes = Proto.encodeCommand protoCommand
      commandDigest = CAS.digestFromBytes commandBytes
  CAS.uploadBlob (RE.recCAS client) commandDigest commandBytes
  
  -- 3. Upload input files and build input root
  inputRootDigest <- uploadInputs client aInputs
  
  -- 4. Build Action and upload to CAS
  let reAction = RE.Action
        { RE.actionCommandDigest = commandDigest
        , RE.actionInputRootDigest = inputRootDigest
        , RE.actionTimeout = Just 3600  -- 1 hour default
        , RE.actionDoNotCache = False
        , RE.actionPlatform = RE.Platform
            { RE.platformProperties = 
                [ RE.PlatformProperty "OSFamily" "Linux"
                , RE.PlatformProperty "container-image" "docker://ghcr.io/straylight-software/nix-worker:latest"
                ]
            }
        }
      -- Upload Action itself to CAS (proper protobuf encoding per REAPI spec)
      protoAction = Proto.ProtoAction
        { Proto.paCommandDigest = CAS.toProtoDigest commandDigest
        , Proto.paInputRootDigest = CAS.toProtoDigest inputRootDigest
        , Proto.paTimeoutSeconds = fromIntegral <$> RE.actionTimeout reAction
        , Proto.paDoNotCache = RE.actionDoNotCache reAction
        , Proto.paPlatform = Just Proto.ProtoPlatform
            { Proto.ppProperties = 
                [ Proto.ProtoPlatformProperty (RE.propName p) (RE.propValue p)
                | p <- RE.platformProperties (RE.actionPlatform reAction)
                ]
            }
        }
      actionBytes = Proto.encodeAction protoAction
      actionDigest = CAS.digestFromBytes actionBytes
  CAS.uploadBlob (RE.recCAS client) actionDigest actionBytes
  
  -- 5. Execute remotely (using Action digest - Action already uploaded to CAS)
  let request = RE.ExecuteRequest
        { RE.erInstanceName = RE.reInstanceName (RE.recConfig client)
        , RE.erActionDigest = actionDigest  -- Pre-computed digest from proper proto serialization
        , RE.erSkipCacheLookup = False
        }
  
  TIO.putStrLn $ "  >> RE: " <> T.intercalate " " (take 3 aCommand) <> " ..."
  result <- RE.executeAndWait client request
  endTime <- getCurrentTime
  
  case result of
    Left err -> pure $ Left err
    Right actionResult -> do
      -- 6. Extract outputs
      let outputPaths = map RE.ofPath (RE.arOutputFiles actionResult)
          exitCode = RE.arExitCode actionResult
      
      if exitCode == 0
        then do
          let proof = Builder.DischargeProof
                { Builder.dpCoeffects = map resourceToCoeffect aCoeffects
                , Builder.dpNetworkAccess = []
                , Builder.dpFilesystemAccess = []
                , Builder.dpAuthUsage = []
                , Builder.dpBuildId = unActionKey $ actionKey action
                , Builder.dpDerivationHash = unActionKey $ actionKey action
                , Builder.dpOutputHashes = []  
                , Builder.dpStartTime = startTime
                , Builder.dpEndTime = endTime
                }
          pure $ Right (outputPaths, proof)
        else 
          pure $ Left $ "Remote execution failed with exit code " <> T.pack (show exitCode)

-- | Upload inputs to CAS and return input root digest
uploadInputs :: RE.REClient -> [ResolvedInput] -> IO CAS.Digest
uploadInputs client inputs = do
  -- Collect file inputs with their contents
  fileInputs <- fmap catMaybes $ forM inputs $ \case
    Resolved_File path -> do
      exists <- doesFileExist path
      if exists
        then do
          content <- BS.readFile path
          pure $ Just (takeFileName path, content)
        else pure Nothing
    Resolved_Store _storePath -> do
      -- For store paths, we assume they're already in the worker's store
      -- or we'd need to upload the entire closure (expensive)
      -- For now, just skip - the worker should have nix store mounted
      pure Nothing
    Resolved_Action _ -> do
      -- Action outputs should have been uploaded by previous execution
      pure Nothing
  
  -- Build input root from collected files
  if null fileInputs
    then do
      -- Empty directory
      let emptyDir = RE.serializeDirectory RE.Directory
            { RE.dirFiles = []
            , RE.dirDirectories = []
            , RE.dirSymlinks = []
            }
          emptyDigest = CAS.digestFromBytes emptyDir
      CAS.uploadBlob (RE.recCAS client) emptyDigest emptyDir
      pure emptyDigest
    else do
      -- Upload each file and build directory
      fileNodes <- forM fileInputs $ \(name, content) -> do
        let digest = CAS.digestFromBytes content
        CAS.uploadBlob (RE.recCAS client) digest content
        -- Check if file is executable (heuristic: .sh files)
        let isExec = ".sh" `T.isSuffixOf` T.pack name
        pure RE.FileNode
          { RE.fnName = T.pack name
          , RE.fnDigest = digest
          , RE.fnIsExecutable = isExec
          }
      
      -- Create root directory
      let rootDir = RE.Directory
            { RE.dirFiles = fileNodes
            , RE.dirDirectories = []
            , RE.dirSymlinks = []
            }
          dirBytes = RE.serializeDirectory rootDir
          dirDigest = CAS.digestFromBytes dirBytes
      
      CAS.uploadBlob (RE.recCAS client) dirDigest dirBytes
      pure dirDigest

-- | Convert Dhall Resource to Builder Coeffect
-- Re-export from Dhall module for convenience
resourceToCoeffect :: Dhall.Resource -> Builder.Coeffect
resourceToCoeffect = Dhall.resourceToCoeffect

-- | Check if coeffects can be satisfied (non-witnessed execution)
-- 
-- For non-witnessed execution, we can only satisfy:
-- - Pure: always ok
-- - Filesystem: check path exists
-- 
-- Network/Auth/Sandbox require witnessed execution to be provable.
checkCoeffects :: [Dhall.Resource] -> IO (Either Dhall.Resource ())
checkCoeffects = go
  where
    go [] = pure $ Right ()
    go (r:rs) = do
      result <- checkOne r
      case result of
        Left missing -> pure $ Left missing
        Right () -> go rs
    
    checkOne :: Dhall.Resource -> IO (Either Dhall.Resource ())
    checkOne = \case
      Dhall.Resource_Pure -> pure $ Right ()
      
      Dhall.Resource_Network -> 
        -- Network without witness is allowed but unattested
        -- The build will work but no proof of what was fetched
        pure $ Right ()
      
      Dhall.Resource_Auth provider -> do
        -- Check for auth token in environment
        -- Convention: PROVIDER_TOKEN (e.g., GITHUB_TOKEN, DOCKER_TOKEN)
        env <- getEnvironment
        let varName = T.unpack $ T.toUpper provider <> "_TOKEN"
        case lookup varName env of
          Just _ -> pure $ Right ()
          Nothing -> pure $ Left (Dhall.Resource_Auth provider)
      
      Dhall.Resource_Sandbox name ->
        -- Sandbox requires explicit setup, can't auto-satisfy
        pure $ Left (Dhall.Resource_Sandbox name)
      
      Dhall.Resource_Filesystem path -> do
        -- Check filesystem path exists
        exists <- doesPathExist (T.unpack path)
        if exists
          then pure $ Right ()
          else pure $ Left (Dhall.Resource_Filesystem path)

-- | Check coeffects for witnessed execution
-- 
-- With witness proxy running, we can satisfy Network coeffect
-- and produce attestations for it.
checkCoeffectsWitnessed :: WitnessConfig -> [Dhall.Resource] -> IO (Either Dhall.Resource ())
checkCoeffectsWitnessed _wc = go
  where
    go [] = pure $ Right ()
    go (r:rs) = do
      result <- checkOne r
      case result of
        Left missing -> pure $ Left missing
        Right () -> go rs
    
    checkOne :: Dhall.Resource -> IO (Either Dhall.Resource ())
    checkOne = \case
      Dhall.Resource_Pure -> pure $ Right ()
      
      Dhall.Resource_Network ->
        -- Witnessed mode: network access will be logged by proxy
        pure $ Right ()
      
      Dhall.Resource_Auth provider -> do
        env <- getEnvironment
        let varName = T.unpack $ T.toUpper provider <> "_TOKEN"
        case lookup varName env of
          Just _ -> pure $ Right ()
          Nothing -> pure $ Left (Dhall.Resource_Auth provider)
      
      Dhall.Resource_Sandbox name ->
        pure $ Left (Dhall.Resource_Sandbox name)
      
      Dhall.Resource_Filesystem path -> do
        exists <- doesPathExist (T.unpack path)
        if exists
          then pure $ Right ()
          else pure $ Left (Dhall.Resource_Filesystem path)

-- -----------------------------------------------------------------------------
-- Local Action Cache (file-based for simplicity)
-- -----------------------------------------------------------------------------

-- | Cache directory for action results
-- Uses XDG_CACHE_HOME or ~/.cache/armitage
actionCacheDir :: IO FilePath
actionCacheDir = do
  env <- getEnvironment
  let home = fromMaybe (error "HOME not set") (lookup "HOME" env)
      cacheBase = fromMaybe (home </> ".cache") (lookup "XDG_CACHE_HOME" env)
  pure $ cacheBase </> "armitage" </> "actions"

-- | Check local cache for action result
checkCAS :: ActionKey -> IO (Maybe [Text])
checkCAS (ActionKey key) = do
  cacheDir <- actionCacheDir
  let cachePath = cacheDir </> T.unpack key
  exists <- doesFileExist cachePath
  if exists
    then do
      content <- TIO.readFile cachePath
      let paths = filter (not . T.null) $ T.lines content
      if null paths
        then pure Nothing
        else pure $ Just paths
    else pure Nothing

-- | Store action result in local cache
storeCAS :: ActionKey -> [Text] -> IO ()
storeCAS (ActionKey key) paths = do
  cacheDir <- actionCacheDir
  createDirectoryIfMissing True cacheDir
  let cachePath = cacheDir </> T.unpack key
      content = T.unlines paths
  TIO.writeFile cachePath content
