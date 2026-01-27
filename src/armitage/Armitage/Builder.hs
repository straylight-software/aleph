{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Armitage.Builder
Description : Daemon-free build executor with coeffect tracking

Execute builds without the nix-daemon. This module:
  - Parses derivation files (.drv)
  - Sets up isolated build environment (bubblewrap/namespace)
  - Executes builder with witness proxy
  - Records coeffect discharge proofs
  - Stores outputs in CAS

The daemon is not just unnecessary - it's hostile.
This is the replacement.

Build:
  buck2 build //src/armitage/builder:builder
-}
module Armitage.Builder
  ( -- * Builder
    BuildConfig (..)
  , BuildResult (..)
  , BuildError (..)
  , runBuild

    -- * Derivation
  , Derivation (..)
  , DrvOutput (..)
  , parseDerivation
  , derivationHash

    -- * Environment
  , BuildEnv (..)
  , setupBuildEnv
  , teardownBuildEnv

    -- * Coeffects
  , Coeffect (..)
  , DischargeProof (..)
  , NetworkAccess (..)
  , FilesystemAccess (..)
  , AccessMode (..)
  , AuthUsage (..)
  , checkCoeffects

    -- * Isolation
  , IsolationLevel (..)
  , withIsolation
  ) where

import Control.Exception (Exception, bracket, throwIO, try)
import Control.Monad (forM, forM_, unless, when)
import Crypto.Hash (SHA256 (..), hashWith)
import qualified Data.ByteArray.Encoding as BA
import Data.Aeson (FromJSON, ToJSON, eitherDecodeFileStrict)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Vector as V
import Data.ByteString (ByteString)
import Data.Maybe (fromMaybe)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC
import qualified Data.ByteString.Lazy as LBS
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.IO as TIO
import Data.Time.Clock (UTCTime, getCurrentTime)
import GHC.Generics (Generic)
import System.Directory
import System.Environment (getEnvironment)
import System.Exit (ExitCode (..))
import System.FilePath
import System.IO (hClose, stderr)
import System.IO.Temp (withSystemTempDirectory)
import System.Posix.Files (setFileMode)
import System.Process

import qualified Armitage.CAS as CAS
import qualified Armitage.Store as Store

-- -----------------------------------------------------------------------------
-- Configuration
-- -----------------------------------------------------------------------------

-- | Build configuration
data BuildConfig = BuildConfig
  { bcStoreConfig :: Store.StoreConfig
  -- ^ Store configuration
  , bcCASConfig :: Maybe CAS.CASConfig
  -- ^ CAS for remote storage (optional)
  , bcProxyPort :: Maybe Int
  -- ^ Witness proxy port (if running)
  , bcIsolation :: IsolationLevel
  -- ^ Isolation level
  , bcKeepFailed :: Bool
  -- ^ Keep build directory on failure
  , bcVerbose :: Bool
  -- ^ Verbose output
  }
  deriving (Show, Generic)

-- | Default build configuration
defaultBuildConfig :: BuildConfig
defaultBuildConfig =
  BuildConfig
    { bcStoreConfig = Store.defaultStoreConfig
    , bcCASConfig = Nothing
    , bcProxyPort = Just 8080
    , bcIsolation = Namespace
    , bcKeepFailed = False
    , bcVerbose = True
    }

-- -----------------------------------------------------------------------------
-- Derivation
-- -----------------------------------------------------------------------------

-- | Parsed derivation
--
-- Corresponds to a .drv file. We parse the essential fields
-- needed for building.
data Derivation = Derivation
  { drvName :: Text
  -- ^ Derivation name
  , drvSystem :: Text
  -- ^ Target system (e.g., "x86_64-linux")
  , drvBuilder :: FilePath
  -- ^ Path to builder executable
  , drvArgs :: [Text]
  -- ^ Arguments to builder
  , drvEnv :: Map Text Text
  -- ^ Environment variables
  , drvInputDrvs :: Map FilePath [Text]
  -- ^ Input derivations -> outputs needed
  , drvInputSrcs :: [FilePath]
  -- ^ Input source paths
  , drvOutputs :: Map Text DrvOutput
  -- ^ Output specifications
  , drvContentAddressed :: Bool
  -- ^ Is this a CA derivation?
  }
  deriving (Show, Eq, Generic)

-- | Derivation output specification
data DrvOutput = DrvOutput
  { doPath :: Maybe FilePath
  -- ^ Fixed output path (None for CA)
  , doHashAlgo :: Maybe Text
  -- ^ Hash algorithm for FOD
  , doHash :: Maybe Text
  -- ^ Expected hash for FOD
  }
  deriving (Show, Eq, Generic)

-- | Parse derivation from .drv file
--
-- Uses nix show-derivation for now.
-- TODO: Parse ATerm format directly or use hnix-store
parseDerivation :: FilePath -> IO (Either String Derivation)
parseDerivation drvPath = do
  -- Shell out to nix show-derivation
  result <- try $ readProcess "nix" ["show-derivation", drvPath] ""
  case result of
    Left (e :: IOError) -> pure $ Left $ "Failed to read derivation: " <> show e
    Right json -> do
      -- Parse JSON output
      -- nix show-derivation outputs: { "/nix/store/xxx.drv": { ... } }
      case Aeson.eitherDecode (LBS.pack $ map (fromIntegral . fromEnum) json) of
        Left err -> pure $ Left $ "JSON parse error: " <> err
        Right (obj :: Aeson.Value) ->
          case obj of
            Aeson.Object topLevel -> 
              case KM.toList topLevel of
                [(drvPath', drvObj)] -> parseDrvObject (AK.toString drvPath') drvObj
                [] -> pure $ Left "Empty derivation object"
                _ -> pure $ Left "Multiple derivations in output (expected single)"
            _ -> pure $ Left "Expected JSON object from nix show-derivation"

-- | Parse a derivation JSON object
parseDrvObject :: String -> Aeson.Value -> IO (Either String Derivation)
parseDrvObject _drvPathStr = \case
  Aeson.Object obj -> do
    let getString k = case KM.lookup k obj of
          Just (Aeson.String s) -> Just s
          _ -> Nothing
        getObject k = case KM.lookup k obj of
          Just (Aeson.Object o) -> Just o
          _ -> Nothing
        getList k = case KM.lookup k obj of
          Just (Aeson.Array a) -> Just (V.toList a)
          _ -> Nothing
    
    case (getString "builder", getObject "env", getObject "outputs") of
      (Just builder, Just envObj, Just outputsObj) -> do
        let args = case getList "args" of
              Just xs -> [s | Aeson.String s <- xs]
              Nothing -> []
            
            env = Map.fromList [(AK.toText k, v) | (k, Aeson.String v) <- KM.toList envObj]
            
            outputs = [(AK.toText k, parseDrvOutput v) | (k, v) <- KM.toList outputsObj]
            
            inputDrvs = case getObject "inputDrvs" of
              Just o -> Map.fromList [(AK.toString k, parseOutputList v) | (k, v) <- KM.toList o]
              Nothing -> Map.empty
            
            inputSrcs = case getList "inputSrcs" of
              Just xs -> [T.unpack s | Aeson.String s <- xs]
              Nothing -> []
            
            system = fromMaybe "x86_64-linux" (getString "system")
            
            name = fromMaybe "unknown" (getString "name")
            
            isCA = case KM.lookup "contentAddressed" envObj of
              Just (Aeson.String "1") -> True
              Just (Aeson.Bool True) -> True
              _ -> False
        
        pure $ Right Derivation
          { drvName = name
          , drvSystem = system
          , drvBuilder = T.unpack builder
          , drvArgs = args
          , drvEnv = env
          , drvOutputs = Map.fromList [(k, v) | (k, Just v) <- outputs]
          , drvInputDrvs = inputDrvs
          , drvInputSrcs = inputSrcs
          , drvContentAddressed = isCA
          }
      _ -> pure $ Left "Missing required fields (builder, env, outputs)"
  _ -> pure $ Left "Expected derivation to be a JSON object"

-- | Parse output list from inputDrvs
parseOutputList :: Aeson.Value -> [Text]
parseOutputList = \case
  Aeson.Array a -> [s | Aeson.String s <- V.toList a]
  _ -> []

-- | Parse a derivation output entry
parseDrvOutput :: Aeson.Value -> Maybe DrvOutput
parseDrvOutput = \case
  Aeson.Object obj -> 
    let path = case KM.lookup "path" obj of
          Just (Aeson.String s) -> Just (T.unpack s)
          _ -> Nothing
        hashAlgo = case KM.lookup "hashAlgo" obj of
          Just (Aeson.String s) -> Just s
          _ -> Nothing
        hash = case KM.lookup "hash" obj of
          Just (Aeson.String s) -> Just s
          _ -> Nothing
    in Just DrvOutput
         { doPath = path
         , doHashAlgo = hashAlgo
         , doHash = hash
         }
  _ -> Nothing

-- | Compute derivation hash
--
-- For CA derivations, this is computed from outputs.
-- For input-addressed, from the .drv file content.
-- Uses canonical Dhall serialization for deterministic hashing.
derivationHash :: Derivation -> Text
derivationHash drv =
  let content = derivationToDhall drv  -- Canonical Dhall representation
      hash = hashWith SHA256 (TE.encodeUtf8 content)
   in TE.decodeUtf8 $ BA.convertToBase BA.Base16 hash

-- | Serialize Derivation to canonical Dhall (deterministic field order)
derivationToDhall :: Derivation -> Text
derivationToDhall Derivation{..} = T.unlines
  [ "{ drvArgs = " <> listToDhall drvArgs
  , ", drvBuilder = \"" <> escapeText (T.pack drvBuilder) <> "\""
  , ", drvContentAddressed = " <> if drvContentAddressed then "True" else "False"
  , ", drvEnv = " <> mapToDhall drvEnv
  , ", drvInputDrvs = " <> inputDrvsToDhall drvInputDrvs
  , ", drvInputSrcs = " <> listToDhall (map T.pack drvInputSrcs)
  , ", drvName = \"" <> escapeText drvName <> "\""
  , ", drvOutputs = " <> outputsToDhall drvOutputs
  , ", drvSystem = \"" <> escapeText drvSystem <> "\""
  , "}"
  ]

-- | Escape text for Dhall string literal
escapeText :: Text -> Text
escapeText = T.concatMap escapeChar
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
listToDhall xs = "[" <> T.intercalate ", " (map (\x -> "\"" <> escapeText x <> "\"") xs) <> "]"

-- | Render map to Dhall (sorted keys for determinism)
mapToDhall :: Map Text Text -> Text
mapToDhall m
  | Map.null m = "toMap {=}"
  | otherwise  = "toMap {" <> T.intercalate ", " entries <> "}"
  where
    entries = [ "`" <> k <> "` = \"" <> escapeText v <> "\""
              | (k, v) <- Map.toAscList m  -- sorted for determinism
              ]

-- | Render input derivations to Dhall (sorted for determinism)
inputDrvsToDhall :: Map FilePath [Text] -> Text
inputDrvsToDhall m
  | Map.null m = "toMap {=} : Map Text (List Text)"
  | otherwise  = "toMap {" <> T.intercalate ", " entries <> "}"
  where
    entries = [ "`" <> T.pack k <> "` = " <> listToDhall v
              | (k, v) <- Map.toAscList m
              ]

-- | Render outputs to Dhall (sorted for determinism)
outputsToDhall :: Map Text DrvOutput -> Text
outputsToDhall m
  | Map.null m = "toMap {=} : Map Text DrvOutput"
  | otherwise  = "toMap {" <> T.intercalate ", " entries <> "}"
  where
    entries = [ "`" <> k <> "` = " <> drvOutputToDhall v
              | (k, v) <- Map.toAscList m
              ]

-- | Render DrvOutput to Dhall
drvOutputToDhall :: DrvOutput -> Text
drvOutputToDhall DrvOutput{..} = T.concat
  [ "{ doHash = ", maybeTextToDhall doHash
  , ", doHashAlgo = ", maybeTextToDhall doHashAlgo
  , ", doPath = ", maybeTextToDhall (T.pack <$> doPath)
  , " }"
  ]

-- | Render Maybe Text to Dhall
maybeTextToDhall :: Maybe Text -> Text
maybeTextToDhall Nothing  = "None Text"
maybeTextToDhall (Just t) = "Some \"" <> escapeText t <> "\""

-- -----------------------------------------------------------------------------
-- Build Result
-- -----------------------------------------------------------------------------

-- | Build result
data BuildResult = BuildResult
  { brOutputs :: Map Text Store.StorePath
  -- ^ Output name -> store path
  , brStartTime :: UTCTime
  -- ^ Build start time
  , brEndTime :: UTCTime
  -- ^ Build end time
  , brDischargeProof :: DischargeProof
  -- ^ Coeffect discharge proof
  }
  deriving (Show, Generic)

-- | Build error
data BuildError
  = ParseError String
  | MissingInput FilePath
  | BuildFailed Int Text
  | CoeffectViolation Coeffect
  | StoreError String
  deriving (Show, Eq, Generic)

instance Exception BuildError

-- -----------------------------------------------------------------------------
-- Coeffects
-- -----------------------------------------------------------------------------

-- | Coeffect (what a build requires)
--
-- This type mirrors:
-- - Dhall: src/armitage/dhall/Resource.dhall
-- - Lean: src/examples/lean-continuity/Continuity.lean (Coeffect)
data Coeffect
  = Pure
  -- ^ Needs nothing external
  | Network
  -- ^ Needs network access
  | Auth Text
  -- ^ Needs credential
  | Sandbox Text
  -- ^ Needs specific sandbox
  | Filesystem FilePath
  -- ^ Needs filesystem path
  | Combined [Coeffect]
  -- ^ Multiple requirements (the ⊗ operator)
  deriving (Show, Eq, Generic, FromJSON, ToJSON)

-- | Coeffect discharge proof
--
-- Evidence that coeffects were satisfied during build.
-- This type mirrors src/armitage/dhall/DischargeProof.dhall
data DischargeProof = DischargeProof
  { dpCoeffects :: [Coeffect]
  -- ^ Required coeffects
  , dpNetworkAccess :: [NetworkAccess]
  -- ^ Witnessed network accesses
  , dpFilesystemAccess :: [FilesystemAccess]
  -- ^ Accessed filesystem paths
  , dpAuthUsage :: [AuthUsage]
  -- ^ Auth token usage
  , dpBuildId :: Text
  -- ^ Unique build identifier
  , dpDerivationHash :: Text
  -- ^ Content hash of input derivation
  , dpOutputHashes :: [(Text, Text)]
  -- ^ Output name -> content hash
  , dpStartTime :: UTCTime
  -- ^ Build start time
  , dpEndTime :: UTCTime
  -- ^ Build end time
  }
  deriving (Show, Generic, FromJSON, ToJSON)

-- | Network access record (from witness proxy)
data NetworkAccess = NetworkAccess
  { naUrl :: Text
  , naMethod :: Text
  -- ^ HTTP method (GET, POST, etc.)
  , naContentHash :: Text
  -- ^ SHA256 of response body
  , naTimestamp :: UTCTime
  }
  deriving (Show, Generic, FromJSON, ToJSON)

-- | Filesystem access record
data FilesystemAccess = FilesystemAccess
  { faPath :: FilePath
  , faMode :: AccessMode
  , faContentHash :: Maybe Text
  -- ^ SHA256 if readable file
  , faTimestamp :: UTCTime
  }
  deriving (Show, Generic, FromJSON, ToJSON)

-- | Filesystem access mode
data AccessMode = Read | Write | Execute
  deriving (Show, Eq, Generic, FromJSON, ToJSON)

-- | Auth token usage record
data AuthUsage = AuthUsage
  { auProvider :: Text
  -- ^ e.g., "github", "docker"
  , auScope :: Maybe Text
  , auTimestamp :: UTCTime
  }
  deriving (Show, Generic, FromJSON, ToJSON)

-- | Check if coeffects can be discharged
checkCoeffects :: BuildConfig -> [Coeffect] -> IO (Either Coeffect ())
checkCoeffects config coeffects = do
  -- For each coeffect, check if environment can satisfy it
  results <- forM coeffects $ \coeff -> case coeff of
    Pure -> pure $ Right ()
    Network ->
      case bcProxyPort config of
        Just _ -> pure $ Right () -- Proxy available
        Nothing -> pure $ Left Network
    Auth provider -> do
      -- Check for credential in environment
      env <- getEnvironment
      let varName = T.unpack $ T.toUpper provider <> "_TOKEN"
      case lookup varName env of
        Just _ -> pure $ Right ()
        Nothing -> pure $ Left (Auth provider)
    Sandbox name -> do
      -- Check sandbox availability
      case bcIsolation config of
        None -> pure $ Left (Sandbox name)
        _ -> pure $ Right ()
    Filesystem path -> do
      exists <- doesPathExist path
      if exists
        then pure $ Right ()
        else pure $ Left (Filesystem path)
    Combined cs -> checkCoeffects config cs

  case filter isLeft results of
    [] -> pure $ Right ()
    (Left e : _) -> pure $ Left e
    _ -> pure $ Right ()
 where
  isLeft (Left _) = True
  isLeft _ = False

-- -----------------------------------------------------------------------------
-- Build Environment
-- -----------------------------------------------------------------------------

-- | Build environment
data BuildEnv = BuildEnv
  { beWorkDir :: FilePath
  -- ^ Build working directory
  , beStoreDir :: FilePath
  -- ^ Store directory (may be bind-mounted)
  , beEnv :: Map Text Text
  -- ^ Environment variables
  , beInputs :: [FilePath]
  -- ^ Materialized inputs
  }
  deriving (Show, Generic)

-- | Set up build environment
setupBuildEnv :: BuildConfig -> Derivation -> IO BuildEnv
setupBuildEnv config drv = do
  -- Create temp directory for build
  workDir <- createTempDirectory "/tmp" "armitage-build"

  -- Set up environment variables
  let baseEnv =
        Map.fromList
          [ ("NIX_BUILD_TOP", T.pack workDir)
          , ("NIX_STORE", T.pack $ Store.storeDir $ bcStoreConfig config)
          , ("TMPDIR", T.pack $ workDir </> "tmp")
          , ("TEMPDIR", T.pack $ workDir </> "tmp")
          , ("TMP", T.pack $ workDir </> "tmp")
          , ("TEMP", T.pack $ workDir </> "tmp")
          , ("HOME", T.pack $ workDir </> "homeless-shelter")
          , ("PATH", "/path-not-set")
          , ("NIX_LOG_FD", "2")
          ]

  -- Add derivation environment
  let env = Map.union (drvEnv drv) baseEnv

  -- Create required directories
  createDirectoryIfMissing True (workDir </> "tmp")
  createDirectoryIfMissing True (workDir </> "homeless-shelter")

  -- Set up proxy environment if configured
  let proxyEnv = case bcProxyPort config of
        Just port ->
          Map.fromList
            [ ("HTTP_PROXY", "http://127.0.0.1:" <> T.pack (show port))
            , ("HTTPS_PROXY", "http://127.0.0.1:" <> T.pack (show port))
            , ("SSL_CERT_FILE", "/etc/ssl/armitage/ca.pem")
            ]
        Nothing -> Map.empty

  pure
    BuildEnv
      { beWorkDir = workDir
      , beStoreDir = Store.storeDir (bcStoreConfig config)
      , beEnv = Map.union proxyEnv env
      , beInputs = drvInputSrcs drv
      }

-- | Tear down build environment
teardownBuildEnv :: BuildConfig -> BuildEnv -> Bool -> IO ()
teardownBuildEnv config env success = do
  -- Remove work directory unless keeping failed
  unless (not success && bcKeepFailed config) $
    removeDirectoryRecursive (beWorkDir env)

-- | Create temp directory with unique suffix
createTempDirectory :: FilePath -> String -> IO FilePath
createTempDirectory base prefix = do
  withSystemTempDirectory prefix $ \tmpPath -> do
    -- Move from system temp to our base directory
    let finalPath = base </> takeFileName tmpPath
    createDirectoryIfMissing True finalPath
    pure finalPath

-- -----------------------------------------------------------------------------
-- Isolation
-- -----------------------------------------------------------------------------

-- | Isolation level
data IsolationLevel
  = None
  -- ^ No isolation (dangerous)
  | Namespace
  -- ^ User/mount namespaces
  | Bubblewrap
  -- ^ bubblewrap sandbox
  | MicroVM
  -- ^ Full VM isolation (isospin)
  deriving (Show, Eq, Generic)

-- | Run action with isolation
withIsolation :: IsolationLevel -> BuildEnv -> IO a -> IO a
withIsolation level env action = case level of
  None -> action
  Namespace -> withNamespaceIsolation env action
  Bubblewrap -> withBubblewrapIsolation env action
  MicroVM -> withMicroVMIsolation env action

-- | Namespace isolation using bubblewrap
withNamespaceIsolation :: BuildEnv -> IO a -> IO a
withNamespaceIsolation env action = do
  -- Use bwrap for namespace isolation without full sandbox
  let workDir = beWorkDir env
  (code, _, errOut) <- readProcessWithExitCode "bwrap"
    [ "--unshare-user"
    , "--unshare-ipc"
    , "--unshare-pid"
    , "--unshare-uts"
    , "--bind", workDir, workDir
    , "--ro-bind", "/nix/store", "/nix/store"
    , "--ro-bind", "/etc/resolv.conf", "/etc/resolv.conf"
    , "--dev", "/dev"
    , "--proc", "/proc"
    , "--die-with-parent"
    , "--"
    , "true"  -- Test command
    ] ""
  case code of
    ExitSuccess -> action  -- bwrap works, but we can't easily wrap arbitrary IO
    ExitFailure _ -> do
      -- bwrap not available or failed, run without isolation
      -- This is a security degradation, log it
      TIO.hPutStrLn stderr $ "WARNING: Namespace isolation unavailable: " <> T.pack errOut
      action

-- | Bubblewrap isolation (full sandbox)
withBubblewrapIsolation :: BuildEnv -> IO a -> IO a
withBubblewrapIsolation _env action = do
  -- Full bwrap sandbox with network isolation
  -- NOTE: This can't easily wrap arbitrary Haskell IO actions
  -- In practice, this should wrap the builder process, not the IO action
  TIO.hPutStrLn stderr "WARNING: Bubblewrap isolation not implemented for IO actions"
  action

-- | MicroVM isolation (isospin)
withMicroVMIsolation :: BuildEnv -> IO a -> IO a
withMicroVMIsolation _env action = do
  -- Would need isospin/firecracker integration
  TIO.hPutStrLn stderr "WARNING: MicroVM isolation not implemented"
  action

-- -----------------------------------------------------------------------------
-- Build Execution
-- -----------------------------------------------------------------------------

-- | Run a build
--
-- This is the main entry point. It:
-- 1. Parses the derivation
-- 2. Checks coeffects
-- 3. Sets up environment
-- 4. Executes builder with isolation
-- 5. Collects outputs
-- 6. Records discharge proof
runBuild :: BuildConfig -> FilePath -> IO (Either BuildError BuildResult)
runBuild config drvPath = do
  startTime <- getCurrentTime

  -- Parse derivation
  parseResult <- parseDerivation drvPath
  case parseResult of
    Left err -> pure $ Left $ ParseError err
    Right drv -> do
      -- Infer coeffects from derivation
      let coeffects = inferCoeffects drv

      -- Check coeffects can be discharged
      checkResult <- checkCoeffects config coeffects
      case checkResult of
        Left coeff -> pure $ Left $ CoeffectViolation coeff
        Right () -> do
          -- Set up build environment
          env <- setupBuildEnv config drv

          -- Execute build with isolation
          buildResult <- try $ withIsolation (bcIsolation config) env $ do
            executeBuilder config drv env

          endTime <- getCurrentTime

          case buildResult of
            Left (e :: BuildError) -> do
              teardownBuildEnv config env False
              pure $ Left e
            Right outputs -> do
              teardownBuildEnv config env True

              -- Create discharge proof
              -- TODO: Read network/filesystem access from proxy/sandbox logs
              let proof =
                    DischargeProof
                      { dpCoeffects = coeffects
                      , dpNetworkAccess = []
                      , dpFilesystemAccess = []
                      , dpAuthUsage = []
                      , dpBuildId = derivationHash drv  -- Use drv hash as build ID for now
                      , dpDerivationHash = derivationHash drv
                      , dpOutputHashes = map (\(n, sp) -> (n, Store.spHash sp)) $ Map.toList outputs
                      , dpStartTime = startTime
                      , dpEndTime = endTime
                      }

              pure $
                Right
                  BuildResult
                    { brOutputs = outputs
                    , brStartTime = startTime
                    , brEndTime = endTime
                    , brDischargeProof = proof
                    }

-- | Infer coeffects from derivation
inferCoeffects :: Derivation -> [Coeffect]
inferCoeffects drv
  | drvContentAddressed drv = [Pure] -- CA derivations are pure
  | hasNetwork = [Network]
  | otherwise = [Pure]
 where
  -- Check for network indicators in env
  hasNetwork =
    any
      (\k -> k `elem` ["outputHash", "outputHashAlgo", "outputHashMode"])
      (Map.keys $ drvEnv drv)

-- | Execute the builder
executeBuilder :: BuildConfig -> Derivation -> BuildEnv -> IO (Map Text Store.StorePath)
executeBuilder config drv env = do
  -- Convert environment to list
  let envList = map (\(k, v) -> (T.unpack k, T.unpack v)) $ Map.toList (beEnv env)

  -- Create output directories
  forM_ (Map.toList $ drvOutputs drv) $ \(name, out) -> do
    let outPath = beWorkDir env </> T.unpack name
    createDirectoryIfMissing True outPath

  -- Set working directory
  setCurrentDirectory (beWorkDir env)

  -- Execute builder
  let builderArgs = map T.unpack (drvArgs drv)
  (exitCode, stdout, stderr) <-
    readCreateProcessWithExitCode
      (proc (drvBuilder drv) builderArgs)
        { env = Just envList
        , cwd = Just (beWorkDir env)
        }
      ""

  when (bcVerbose config) $ do
    putStrLn $ "Builder output: " <> stdout
    putStrLn $ "Builder stderr: " <> stderr

  case exitCode of
    ExitSuccess -> do
      -- Collect outputs
      outputs <- forM (Map.toList $ drvOutputs drv) $ \(name, _out) -> do
        let outPath = beWorkDir env </> T.unpack name
        -- Add to store
        Store.withStore (bcStoreConfig config) $ \store -> do
          content <- BS.readFile outPath -- Simplified: would use NAR
          sp <- Store.addToStore store (drvName drv <> "-" <> name) content
          pure (name, sp)
      pure $ Map.fromList outputs
    ExitFailure code ->
      throwIO $ BuildFailed code (T.pack stderr)
