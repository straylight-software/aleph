{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Armitage.RE
Description : Remote Execution API client for NativeLink

Client for the Remote Execution API Execution service.
This is where DICE meets NativeLink - actions get sent to remote workers.

The flow:
  1. Upload inputs to CAS (files, toolchain blobs)
  2. Create Action (command + input tree + platform)
  3. Execute via RE API (returns operation name)
  4. Poll for completion or stream updates
  5. Fetch outputs from CAS

We use the standard REAPI v2:
  https://github.com/bazelbuild/remote-apis/blob/main/build/bazel/remote/execution/v2/remote_execution.proto
-}
module Armitage.RE (
    -- * Configuration
    REConfig (..),
    defaultConfig,
    nativeLinkConfig,

    -- * Client
    REClient (..),
    withREClient,

    -- * Types
    Command (..),
    Platform (..),
    PlatformProperty (..),
    Action (..),
    ActionResult (..),
    OutputFile (..),
    ExecuteRequest (..),
    ExecuteResponse (..),

    -- * Execution
    execute,
    executeAndWait,
    getActionResult,

    -- * Directory operations
    Directory (..),
    FileNode (..),
    DirectoryNode (..),
    uploadDirectory,
    computeInputRoot,
) where

import Control.Concurrent (threadDelay)
import Control.Exception (bracket)
import Control.Monad (forM, forM_, when)
import Crypto.Hash (SHA256 (..), hashWith)
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Time.Clock (UTCTime, getCurrentTime)
import GHC.Generics (Generic)
import Network.Socket (PortNumber)
import System.Directory (doesFileExist, getFileSize)
import System.FilePath (takeFileName)

import Network.GRPC.Client (
    Address (..),
    ConnParams (..),
    Connection,
    Server (..),
    ServerValidation (..),
    certStoreFromSystem,
    withConnection,
 )
import Network.GRPC.Common (def)

import qualified Armitage.CAS as CAS

-- -----------------------------------------------------------------------------
-- Configuration
-- -----------------------------------------------------------------------------

-- | RE client configuration
data REConfig = REConfig
    { reHost :: String
    -- ^ Execution server hostname
    , rePort :: PortNumber
    -- ^ Execution server port
    , reUseTLS :: Bool
    -- ^ Use TLS
    , reInstanceName :: Text
    -- ^ Instance name (usually "main")
    , reCASConfig :: CAS.CASConfig
    -- ^ CAS config (often same endpoint)
    }
    deriving (Show, Eq, Generic)

-- | Default configuration for local NativeLink
defaultConfig :: REConfig
defaultConfig =
    REConfig
        { reHost = "localhost"
        , rePort = 50051
        , reUseTLS = False
        , reInstanceName = "main"
        , reCASConfig = CAS.defaultConfig
        }

-- | Configuration for NativeLink on Fly.io
nativeLinkConfig :: REConfig
nativeLinkConfig =
    REConfig
        { reHost = "aleph-scheduler.fly.dev"
        , rePort = 443
        , reUseTLS = True
        , reInstanceName = "main"
        , reCASConfig = CAS.flyConfig
        }

-- -----------------------------------------------------------------------------
-- Types (matching REAPI protos)
-- -----------------------------------------------------------------------------

-- | Platform requirements for execution
data Platform = Platform
    { platformProperties :: [PlatformProperty]
    }
    deriving (Show, Eq, Generic)

-- | A platform property (key-value)
data PlatformProperty = PlatformProperty
    { propName :: Text
    , propValue :: Text
    }
    deriving (Show, Eq, Generic)

-- | Command to execute
data Command = Command
    { cmdArguments :: [Text]
    -- ^ Command line arguments (first is executable)
    , cmdEnvironmentVariables :: [(Text, Text)]
    -- ^ Environment variables
    , cmdOutputFiles :: [Text]
    -- ^ Expected output file paths (relative to working dir)
    , cmdOutputDirectories :: [Text]
    -- ^ Expected output directory paths
    , cmdWorkingDirectory :: Text
    -- ^ Working directory (relative to input root)
    , cmdOutputPaths :: [Text]
    -- ^ Combined output paths (REAPI v2.1+)
    }
    deriving (Show, Eq, Generic)

-- | An action to execute
data Action = Action
    { actionCommandDigest :: CAS.Digest
    -- ^ Digest of serialized Command in CAS
    , actionInputRootDigest :: CAS.Digest
    -- ^ Digest of input Directory tree root
    , actionTimeout :: Maybe Int
    -- ^ Timeout in seconds
    , actionDoNotCache :: Bool
    -- ^ Skip action cache
    , actionPlatform :: Platform
    -- ^ Platform requirements
    }
    deriving (Show, Eq, Generic)

-- | Result of an action execution
data ActionResult = ActionResult
    { arOutputFiles :: [OutputFile]
    -- ^ Output files produced
    , arOutputDirectories :: [CAS.Digest]
    -- ^ Output directory tree digests
    , arExitCode :: Int
    -- ^ Exit code of the command
    , arStdoutDigest :: Maybe CAS.Digest
    -- ^ Stdout blob digest (if captured)
    , arStderrDigest :: Maybe CAS.Digest
    -- ^ Stderr blob digest (if captured)
    , arExecutionMetadata :: ExecutionMetadata
    -- ^ Timing and worker info
    }
    deriving (Show, Eq, Generic)

-- | An output file
data OutputFile = OutputFile
    { ofPath :: Text
    -- ^ Path relative to working directory
    , ofDigest :: CAS.Digest
    -- ^ Content digest
    , ofIsExecutable :: Bool
    -- ^ Was marked executable
    }
    deriving (Show, Eq, Generic)

-- | Execution metadata
data ExecutionMetadata = ExecutionMetadata
    { emWorker :: Text
    -- ^ Worker that executed
    , emQueuedTime :: Maybe UTCTime
    , emWorkerStartTime :: Maybe UTCTime
    , emWorkerCompletedTime :: Maybe UTCTime
    , emInputFetchStartTime :: Maybe UTCTime
    , emInputFetchCompletedTime :: Maybe UTCTime
    , emExecutionStartTime :: Maybe UTCTime
    , emExecutionCompletedTime :: Maybe UTCTime
    , emOutputUploadStartTime :: Maybe UTCTime
    , emOutputUploadCompletedTime :: Maybe UTCTime
    }
    deriving (Show, Eq, Generic)

-- | Execute request
data ExecuteRequest = ExecuteRequest
    { erInstanceName :: Text
    , erAction :: Action
    , erSkipCacheLookup :: Bool
    }
    deriving (Show, Eq, Generic)

-- | Execute response (simplified - real one is Operation with metadata)
data ExecuteResponse = ExecuteResponse
    { exResult :: Maybe ActionResult
    -- ^ Result if completed
    , exOperationName :: Text
    -- ^ Operation name for polling
    , exDone :: Bool
    -- ^ Whether execution is complete
    , exError :: Maybe Text
    -- ^ Error message if failed
    }
    deriving (Show, Eq, Generic)

-- -----------------------------------------------------------------------------
-- Directory tree types
-- -----------------------------------------------------------------------------

-- | A directory in the Merkle tree
data Directory = Directory
    { dirFiles :: [FileNode]
    , dirDirectories :: [DirectoryNode]
    , dirSymlinks :: [SymlinkNode]
    }
    deriving (Show, Eq, Generic)

-- | A file in a directory
data FileNode = FileNode
    { fnName :: Text
    -- ^ File name (not path)
    , fnDigest :: CAS.Digest
    -- ^ Content digest
    , fnIsExecutable :: Bool
    }
    deriving (Show, Eq, Generic)

-- | A subdirectory reference
data DirectoryNode = DirectoryNode
    { dnName :: Text
    -- ^ Directory name (not path)
    , dnDigest :: CAS.Digest
    -- ^ Digest of Directory message
    }
    deriving (Show, Eq, Generic)

-- | A symlink
data SymlinkNode = SymlinkNode
    { snName :: Text
    , snTarget :: Text
    }
    deriving (Show, Eq, Generic)

-- -----------------------------------------------------------------------------
-- Client
-- -----------------------------------------------------------------------------

-- | Opaque RE client handle
data REClient = REClient
    { recConfig :: REConfig
    , recConn :: Connection
    , recCAS :: CAS.CASClient
    }

-- | Create RE client and run action
withREClient :: REConfig -> (REClient -> IO a) -> IO a
withREClient config action = do
    let addr =
            Address
                { addressHost = reHost config
                , addressPort = rePort config
                , addressAuthority = Nothing
                }
        serverValidation = ValidateServer certStoreFromSystem
        server =
            if reUseTLS config
                then ServerSecure serverValidation def addr
                else ServerInsecure addr
        params = def

    withConnection params server $ \conn ->
        CAS.withCASClient (reCASConfig config) $ \casClient ->
            action
                REClient
                    { recConfig = config
                    , recConn = conn
                    , recCAS = casClient
                    }

-- -----------------------------------------------------------------------------
-- Execution
-- -----------------------------------------------------------------------------

-- | Execute an action (non-blocking, returns operation)
execute :: REClient -> ExecuteRequest -> IO ExecuteResponse
execute client req = do
    -- TODO: actual gRPC call to Execution.Execute
    -- This is a streaming RPC that returns Operation updates
    putStrLn $ "RE: execute action " <> T.unpack (CAS.digestHash $ actionCommandDigest $ erAction req)
    pure
        ExecuteResponse
            { exResult = Nothing
            , exOperationName = "operations/pending-" <> CAS.digestHash (actionCommandDigest $ erAction req)
            , exDone = False
            , exError = Nothing
            }

-- | Execute and wait for completion (blocking)
executeAndWait :: REClient -> ExecuteRequest -> IO (Either Text ActionResult)
executeAndWait client req = do
    putStrLn $ "RE: executeAndWait"

    -- 1. Start execution
    response <- execute client req

    -- 2. Poll until done
    let poll opName = do
            threadDelay 100000 -- 100ms
            -- TODO: call Operations.GetOperation
            -- For now, simulate completion
            pure $
                Right
                    ActionResult
                        { arOutputFiles = []
                        , arOutputDirectories = []
                        , arExitCode = 0
                        , arStdoutDigest = Nothing
                        , arStderrDigest = Nothing
                        , arExecutionMetadata =
                            ExecutionMetadata
                                { emWorker = "local-stub"
                                , emQueuedTime = Nothing
                                , emWorkerStartTime = Nothing
                                , emWorkerCompletedTime = Nothing
                                , emInputFetchStartTime = Nothing
                                , emInputFetchCompletedTime = Nothing
                                , emExecutionStartTime = Nothing
                                , emExecutionCompletedTime = Nothing
                                , emOutputUploadStartTime = Nothing
                                , emOutputUploadCompletedTime = Nothing
                                }
                        }

    if exDone response
        then case exResult response of
            Just result -> pure $ Right result
            Nothing -> pure $ Left $ fromMaybe "Unknown error" (exError response)
        else poll (exOperationName response)
  where
    fromMaybe d Nothing = d
    fromMaybe _ (Just x) = x

-- | Get cached action result (if exists)
getActionResult :: REClient -> CAS.Digest -> IO (Maybe ActionResult)
getActionResult client actionDigest = do
    -- TODO: call ActionCache.GetActionResult
    putStrLn $ "RE: getActionResult " <> T.unpack (CAS.digestHash actionDigest)
    pure Nothing

-- -----------------------------------------------------------------------------
-- Directory operations
-- -----------------------------------------------------------------------------

-- | Upload a directory tree to CAS, return root digest
uploadDirectory :: REClient -> FilePath -> IO CAS.Digest
uploadDirectory client path = do
    -- TODO: walk directory, build Merkle tree, upload all blobs
    putStrLn $ "RE: uploadDirectory " <> path
    pure $ CAS.Digest "stubhash" 0

-- | Compute input root from list of files
computeInputRoot :: REClient -> [(FilePath, ByteString)] -> IO CAS.Digest
computeInputRoot client files = do
    -- Upload each file to CAS
    forM_ files $ \(path, content) -> do
        let digest = CAS.digestFromBytes content
        CAS.uploadBlob (recCAS client) digest content

    -- Build directory tree
    let fileNodes = flip map files $ \(path, content) ->
            FileNode
                { fnName = T.pack $ takeFileName path
                , fnDigest = CAS.digestFromBytes content
                , fnIsExecutable = False
                }
        rootDir =
            Directory
                { dirFiles = fileNodes
                , dirDirectories = []
                , dirSymlinks = []
                }

    -- Serialize and upload directory
    let dirBytes = serializeDirectory rootDir
        dirDigest = CAS.digestFromBytes dirBytes
    CAS.uploadBlob (recCAS client) dirDigest dirBytes

    pure dirDigest

-- | Serialize a Directory (stub - would use protobuf)
serializeDirectory :: Directory -> ByteString
serializeDirectory dir =
    -- TODO: proper protobuf serialization
    BC.pack $ show dir
