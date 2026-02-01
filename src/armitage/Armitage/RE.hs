{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

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
    SymlinkNode (..),
    uploadDirectory,
    computeInputRoot,
    serializeDirectory,
) where

import Control.Monad (foldM, forM, forM_)
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import Data.Time.Clock (UTCTime)
import GHC.Generics (Generic)
import Network.Socket (PortNumber)
import System.Directory (doesDirectoryExist, executable, getPermissions, listDirectory)
import System.FilePath (takeFileName, (</>))

import qualified Data.Bits
import Data.Proxy (Proxy (..))
import Network.GRPC.Client (
    Address (..),
    Connection,
    Server (..),
    ServerValidation (..),
    certStoreFromSystem,
    recvFinalOutput,
    recvNextOutputElem,
    sendFinalInput,
    withConnection,
    withRPC,
 )
import Network.GRPC.Common (NextElem (..), def)

import qualified Armitage.CAS as CAS
import qualified Armitage.Proto as Proto

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

{- | Execute request
Note: Action must already be uploaded to CAS before calling execute.
Pass the digest of the uploaded Action, not the Action itself.
-}
data ExecuteRequest = ExecuteRequest
    { erInstanceName :: Text
    , erActionDigest :: CAS.Digest -- Digest of Action already in CAS
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

{- | Execute an action (non-blocking, returns operation)
Uses Execution.Execute RPC which is server-streaming (returns Operations)
NOTE: The Action must already be uploaded to CAS before calling this.
-}
execute :: REClient -> ExecuteRequest -> IO ExecuteResponse
execute client req = do
    let
        -- Use the pre-computed digest (Action already in CAS)
        protoActionDigest = CAS.toProtoDigest (erActionDigest req)
        protoReq =
            Proto.ExecuteRequest
                { Proto.exrInstanceName = reInstanceName (recConfig client)
                , Proto.exrActionDigest = protoActionDigest
                , Proto.exrSkipCacheLookup = erSkipCacheLookup req
                }

    -- Server-streaming RPC: send request, receive stream of Operations
    withRPC (recConn client) def (Proxy @Proto.ExecutionExecute) $ \call -> do
        sendFinalInput call (Proto.encodeExecuteRequest protoReq)

        -- Collect operations until we get done=true or stream ends
        let collectOps = do
                nextElem <- recvNextOutputElem call
                case nextElem of
                    NoNextElem ->
                        pure
                            ExecuteResponse
                                { exResult = Nothing
                                , exOperationName = ""
                                , exDone = False
                                , exError = Just "Stream ended without completion"
                                }
                    NextElem opBytes -> case Proto.decodeOperation opBytes of
                        Nothing ->
                            pure
                                ExecuteResponse
                                    { exResult = Nothing
                                    , exOperationName = ""
                                    , exDone = False
                                    , exError = Just "Failed to decode Operation"
                                    }
                        Just op -> do
                            if Proto.opDone op
                                then do
                                    -- Extract ActionResult from the response if present
                                    let result = Proto.opResponse op >>= decodeExecuteResponseResult
                                        errMsg =
                                            if Proto.opErrorCode op /= 0
                                                then Proto.opErrorMessage op
                                                else Nothing
                                    pure
                                        ExecuteResponse
                                            { exResult = result
                                            , exOperationName = Proto.opName op
                                            , exDone = True
                                            , exError = errMsg
                                            }
                                else collectOps -- Keep reading until done
        collectOps

{- | Execute and wait for completion (blocking)
Uses the streaming response from Execute - it continues until done=true
-}
executeAndWait :: REClient -> ExecuteRequest -> IO (Either Text ActionResult)
executeAndWait client req = do
    -- Execute already streams until completion due to collectOps recursion
    response <- execute client req

    if exDone response
        then case exResult response of
            Just result -> pure $ Right result
            Nothing -> pure $ Left $ fromMaybe "Execution failed (no result)" (exError response)
        else
            -- If execute returned without done=true, the stream ended prematurely
            pure $ Left "Execution stream ended before completion"

{- | Get cached action result (if exists)
Uses ActionCache.GetActionResult RPC (non-streaming)
-}
getActionResult :: REClient -> CAS.Digest -> IO (Maybe ActionResult)
getActionResult client actionDigest = do
    let protoDigest =
            Proto.ProtoDigest
                { Proto.pdHash = CAS.digestHash actionDigest
                , Proto.pdSizeBytes = fromIntegral (CAS.digestSize actionDigest)
                }
        protoReq =
            Proto.GetActionResultRequest
                { Proto.garrInstanceName = reInstanceName (recConfig client)
                , Proto.garrActionDigest = protoDigest
                }

    result <- withRPC (recConn client) def (Proxy @Proto.ActionCacheGetActionResult) $ \call -> do
        sendFinalInput call (Proto.encodeGetActionResultRequest protoReq)
        (respBytes, _trailers) <- recvFinalOutput call
        pure $ Proto.decodeActionResult (LBS.toStrict respBytes) >>= protoToActionResult
    pure result

-- -----------------------------------------------------------------------------
-- Directory operations
-- -----------------------------------------------------------------------------

{- | Upload a directory tree to CAS, return root digest
Walks the directory recursively, uploads all files, builds Merkle tree
-}
uploadDirectory :: REClient -> FilePath -> IO CAS.Digest
uploadDirectory client rootPath = do
    uploadDirRecursive client rootPath
  where
    uploadDirRecursive :: REClient -> FilePath -> IO CAS.Digest
    uploadDirRecursive cli dirPath = do
        entries <- listDirectory dirPath

        -- Separate files and directories
        (files, dirs) <- foldM (categorize dirPath) ([], []) entries

        -- Upload files and create FileNodes
        fileNodes <- forM files $ \filePath -> do
            content <- BS.readFile filePath
            let digest = CAS.digestFromBytes content
            CAS.uploadBlob (recCAS cli) digest content
            isExec <- isExecutable filePath
            pure
                FileNode
                    { fnName = T.pack (takeFileName filePath)
                    , fnDigest = digest
                    , fnIsExecutable = isExec
                    }

        -- Recursively upload subdirectories and create DirectoryNodes
        dirNodes <- forM dirs $ \subDirPath -> do
            subDigest <- uploadDirRecursive cli subDirPath
            pure
                DirectoryNode
                    { dnName = T.pack (takeFileName subDirPath)
                    , dnDigest = subDigest
                    }

        -- Create and upload this directory
        let dir =
                Directory
                    { dirFiles = fileNodes
                    , dirDirectories = dirNodes
                    , dirSymlinks = [] -- TODO: handle symlinks
                    }
            dirBytes = serializeDirectory dir
            dirDigest = CAS.digestFromBytes dirBytes

        CAS.uploadBlob (recCAS cli) dirDigest dirBytes
        pure dirDigest

    categorize :: FilePath -> ([FilePath], [FilePath]) -> FilePath -> IO ([FilePath], [FilePath])
    categorize parent (fs, ds) name = do
        let fullPath = parent </> name
        isDir <- doesDirectoryExist fullPath
        if isDir
            then pure (fs, fullPath : ds)
            else pure (fullPath : fs, ds)

    isExecutable :: FilePath -> IO Bool
    isExecutable fp = do
        perms <- getPermissions fp
        pure $ executable perms

-- | Compute input root from list of files
computeInputRoot :: REClient -> [(FilePath, ByteString)] -> IO CAS.Digest
computeInputRoot client files = do
    -- Upload each file to CAS
    forM_ files $ \(_path, content) -> do
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

-- | Serialize a Directory to protobuf wire format
serializeDirectory :: Directory -> ByteString
serializeDirectory dir = Proto.encodeDirectory protoDir
  where
    protoDir =
        Proto.ProtoDirectory
            { Proto.pdFiles = map toProtoFileNode (dirFiles dir)
            , Proto.pdDirectories = map toProtoDirectoryNode (dirDirectories dir)
            , Proto.pdSymlinks = map toProtoSymlinkNode (dirSymlinks dir)
            }

    toProtoFileNode fn =
        Proto.ProtoFileNode
            { Proto.pfnName = fnName fn
            , Proto.pfnDigest = toProtoDigest (fnDigest fn)
            , Proto.pfnIsExecutable = fnIsExecutable fn
            }

    toProtoDirectoryNode dn =
        Proto.ProtoDirectoryNode
            { Proto.pdnName = dnName dn
            , Proto.pdnDigest = toProtoDigest (dnDigest dn)
            }

    toProtoSymlinkNode sn =
        Proto.ProtoSymlinkNode
            { Proto.psnName = snName sn
            , Proto.psnTarget = snTarget sn
            }

    toProtoDigest d =
        Proto.ProtoDigest
            { Proto.pdHash = CAS.digestHash d
            , Proto.pdSizeBytes = fromIntegral (CAS.digestSize d)
            }

-- -----------------------------------------------------------------------------
-- Proto to domain type conversions
-- -----------------------------------------------------------------------------

-- | Convert ProtoDigest to CAS.Digest
fromProtoDigest :: Proto.ProtoDigest -> CAS.Digest
fromProtoDigest pd =
    CAS.Digest
        { CAS.digestHash = Proto.pdHash pd
        , CAS.digestSize = fromIntegral (Proto.pdSizeBytes pd)
        }

-- | Convert ProtoActionResult to ActionResult
protoToActionResult :: Proto.ProtoActionResult -> Maybe ActionResult
protoToActionResult par =
    Just
        ActionResult
            { arOutputFiles = map toOutputFile (Proto.parOutputFiles par)
            , arOutputDirectories = [] -- Not decoded yet
            , arExitCode = Proto.parExitCode par
            , arStdoutDigest = fromProtoDigest <$> Proto.parStdoutDigest par
            , arStderrDigest = fromProtoDigest <$> Proto.parStderrDigest par
            , arExecutionMetadata = emptyMetadata
            }
  where
    toOutputFile (path, digest, executable) =
        OutputFile
            { ofPath = path
            , ofDigest = fromProtoDigest digest
            , ofIsExecutable = executable
            }
    emptyMetadata =
        ExecutionMetadata
            { emWorker = ""
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

{- | Decode ActionResult from ExecuteResponse's Any field
The Any message has type_url (field 1) and value (field 2)
For ExecuteResponse, the value contains an ActionResult
-}
decodeExecuteResponseResult :: ByteString -> Maybe ActionResult
decodeExecuteResponseResult anyBytes = do
    -- Parse the Any message to extract the value field
    let fields = parseAnyFields anyBytes
    valueField <- lookup 2 fields -- field 2 is the serialized message
    -- The value contains an ExecuteResponse, which has result in field 1
    let execRespFields = parseAnyFields valueField
    resultField <- lookup 1 execRespFields -- field 1 is ActionResult
    Proto.decodeActionResult resultField >>= protoToActionResult
  where
    -- Simple field parser using Proto module's varint functions
    parseAnyFields :: ByteString -> [(Int, ByteString)]
    parseAnyFields bs
        | BS.null bs = []
        | otherwise = case parseField bs of
            Nothing -> []
            Just (fieldNum, content, rest) ->
                (fieldNum, content) : parseAnyFields rest

    parseField :: ByteString -> Maybe (Int, ByteString, ByteString)
    parseField bs = do
        (key, rest1) <- Proto.decodeVarint bs
        let fieldNum = fromIntegral (key `Data.Bits.shiftR` 3)
            wireType = key Data.Bits..&. 0x07
        case wireType of
            0 -> do
                -- varint - encode it back for lookup
                (val, rest2) <- Proto.decodeVarint rest1
                Just (fieldNum, Proto.encodeVarint val, rest2)
            2 -> do
                -- length-delimited
                (len, rest2) <- Proto.decodeVarint rest1
                let (content, rest3) = BS.splitAt (fromIntegral len) rest2
                Just (fieldNum, content, rest3)
            _ -> Nothing
