{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}

{- |
Module      : Armitage.CAS
Description : Content-Addressed Storage client for NativeLink

Client for the Remote Execution API ContentAddressableStorage service.
Uses grapesy for gRPC with hand-written protobuf encoding (no proto-lens).

The CAS stores blobs by their SHA256 digest. This module provides:
  - Upload blobs (batch for small, streaming for large)
  - Download blobs (via ByteStream)
  - Check for missing blobs
  - Digest computation

All operations use the standard Remote Execution API:
  https://github.com/bazelbuild/remote-apis
-}
module Armitage.CAS
  ( -- * Configuration
    CASConfig (..)
  , defaultConfig
  , flyConfig

    -- * Client
  , CASClient
  , withCASClient

    -- * Digest
  , Digest (..)
  , digestFromBytes
  , digestToResourceName

    -- * Operations
  , uploadBlob
  , downloadBlob
  , findMissingBlobs
  , blobExists

    -- * Utilities
  , hashBytes
  ) where

import Control.Exception (try, SomeException)
import System.IO (hPutStrLn, hFlush, stderr)
import Control.Monad (forM_)
import Crypto.Hash (SHA256 (..), hashWith)
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import Data.Proxy (Proxy(..))
import Data.Text (Text)
import qualified Data.Text as T
import GHC.Generics (Generic)
import Network.Socket (PortNumber)

-- grapesy imports
import Network.GRPC.Client
  ( Connection
  , Server (..)
  , Address (..)
  , ServerValidation (..)
  , certStoreFromSystem
  , withConnection
  , withRPC
  , sendFinalInput
  , recvFinalOutput
  , sendNextInput
  , sendEndOfInput
  , recvNextOutputElem
  )
import Network.GRPC.Common (def, NextElem(..))

-- Our hand-written protobuf types
import qualified Armitage.Proto as Proto

-- -----------------------------------------------------------------------------
-- Configuration
-- -----------------------------------------------------------------------------

-- | CAS client configuration
data CASConfig = CASConfig
  { casHost :: String
  -- ^ CAS server hostname (e.g., "localhost" or "cas.straylight.cx")
  , casPort :: PortNumber
  -- ^ CAS server port (e.g., 50052 for local, 443 for TLS)
  , casUseTLS :: Bool
  -- ^ Use TLS (required for production endpoints)
  , casInstanceName :: Text
  -- ^ Remote execution instance name (usually "main")
  }
  deriving (Show, Eq, Generic)

-- | Default configuration for local NativeLink
-- NativeLink runs all services (CAS, Execution, ActionCache) on the same port
defaultConfig :: CASConfig
defaultConfig =
  CASConfig
    { casHost = "localhost"
    , casPort = 50051
    , casUseTLS = False
    , casInstanceName = "main"
    }

-- | Configuration for Fly.io deployment
flyConfig :: CASConfig
flyConfig =
  CASConfig
    { casHost = "aleph-cas.fly.dev"
    , casPort = 443
    , casUseTLS = True
    , casInstanceName = "main"
    }

-- -----------------------------------------------------------------------------
-- Digest
-- -----------------------------------------------------------------------------

-- | Content digest (hash + size)
--
-- This matches the Remote Execution API Digest message:
--   message Digest {
--     string hash = 1;      // SHA256 hex
--     int64 size_bytes = 2;
--   }
data Digest = Digest
  { digestHash :: Text
  -- ^ SHA256 hash as lowercase hex string (64 chars)
  , digestSize :: Int
  -- ^ Size in bytes
  }
  deriving (Show, Eq, Generic)

-- | Compute digest from bytes
digestFromBytes :: ByteString -> Digest
digestFromBytes bs =
  Digest
    { digestHash = hashBytes bs
    , digestSize = BS.length bs
    }

-- | Convert digest to ByteStream resource name
--
-- Format: {instance_name}/blobs/{hash}/{size}
digestToResourceName :: Text -> Digest -> Text
digestToResourceName instanceName Digest {..} =
  instanceName <> "/blobs/" <> digestHash <> "/" <> T.pack (show digestSize)

-- | Hash bytes to SHA256 hex string
hashBytes :: ByteString -> Text
hashBytes bs = T.pack $ show $ hashWith SHA256 bs

-- | Convert our Digest to Proto Digest
toProtoDigest :: Digest -> Proto.ProtoDigest
toProtoDigest Digest{..} = Proto.ProtoDigest
  { Proto.pdHash = digestHash
  , Proto.pdSizeBytes = fromIntegral digestSize
  }

-- | Convert Proto Digest to our Digest
fromProtoDigest :: Proto.ProtoDigest -> Digest
fromProtoDigest Proto.ProtoDigest{..} = Digest
  { digestHash = pdHash
  , digestSize = fromIntegral pdSizeBytes
  }

-- -----------------------------------------------------------------------------
-- Client
-- -----------------------------------------------------------------------------

-- | Opaque CAS client handle
data CASClient = CASClient
  { clientConfig :: CASConfig
  , clientConn :: Connection
  }

-- | Create CAS client and run action
--
-- Example:
--   withCASClient config $ \client -> do
--     let digest = digestFromBytes content
--     uploadBlob client digest content
withCASClient :: CASConfig -> (CASClient -> IO a) -> IO a
withCASClient config action = do
  hPutStrLn stderr $ "CAS: connecting to " <> casHost config <> ":" <> show (casPort config)
  hFlush stderr
  let addr = Address
        { addressHost = casHost config
        , addressPort = casPort config
        , addressAuthority = Nothing
        }
      -- Use system certificate store for TLS validation
      -- NoServerValidation would skip verification (insecure)
      serverValidation = ValidateServer certStoreFromSystem
      server = if casUseTLS config
        then ServerSecure serverValidation def addr
        else ServerInsecure addr
      params = def
  hPutStrLn stderr $ "CAS: server config = " <> show (casUseTLS config)
  hFlush stderr
  withConnection params server $ \conn -> do
    hPutStrLn stderr "CAS: connection established"
    hFlush stderr
    action CASClient
      { clientConfig = config
      , clientConn = conn
      }

-- -----------------------------------------------------------------------------
-- Operations
-- -----------------------------------------------------------------------------

-- | Upload blob to CAS
--
-- Uses BatchUpdateBlobs for small blobs (<4MB),
-- ByteStream.Write for large blobs.
uploadBlob :: CASClient -> Digest -> ByteString -> IO ()
uploadBlob client digest content
  | BS.length content < 4 * 1024 * 1024 = batchUpload client digest content
  | otherwise = streamUpload client digest content

-- | Batch upload (for blobs < 4MB)
batchUpload :: CASClient -> Digest -> ByteString -> IO ()
batchUpload client digest content = do
  let request = Proto.BatchUpdateBlobsRequest
        { Proto.bubrInstanceName = casInstanceName (clientConfig client)
        , Proto.bubrRequests =
            [ Proto.BlobRequest
                { Proto.brDigest = toProtoDigest digest
                , Proto.brData = content
                }
            ]
        }
      reqBytes = Proto.encodeBatchUpdateBlobsRequest request

  result <- try @SomeException $
    withRPC (clientConn client) def (Proxy @Proto.CASBatchUpdateBlobs) $ \call -> do
      sendFinalInput call reqBytes
      (respBytes, _trailers) <- recvFinalOutput call
      pure respBytes

  case result of
    Left err -> do
      putStrLn $ "CAS: batch upload failed: " <> show err
    Right respBytes -> do
      case Proto.decodeBatchUpdateBlobsResponse respBytes of
        Nothing ->
          putStrLn $ "CAS: batch upload response decode failed"
        Just resp -> do
          let responses = Proto.bubrResponses resp
          forM_ responses $ \r -> do
            if Proto.brespStatusCode r == 0
              then putStrLn $ "CAS: uploaded " <> T.unpack (Proto.pdHash (Proto.brespDigest r))
              else putStrLn $ "CAS: upload failed with status " <> show (Proto.brespStatusCode r)

-- | Stream upload (for blobs >= 4MB)
streamUpload :: CASClient -> Digest -> ByteString -> IO ()
streamUpload client digest content = do
  let resourceName = casInstanceName (clientConfig client)
        <> "/uploads/armitage-" <> T.pack (show $ BS.length content)
        <> "/blobs/" <> digestHash digest
        <> "/" <> T.pack (show (digestSize digest))

      -- Split content into 1MB chunks
      chunkSize = 1024 * 1024
      chunks = chunksOf chunkSize content

      makeRequest offset isLast chunk = Proto.WriteRequest
        { Proto.wrResourceName = resourceName
        , Proto.wrWriteOffset = offset
        , Proto.wrFinishWrite = isLast
        , Proto.wrData = chunk
        }

  result <- try @SomeException $
    withRPC (clientConn client) def (Proxy @Proto.ByteStreamWrite) $ \call -> do
      -- Send all chunks
      let sendChunks [] _ = sendEndOfInput call
          sendChunks [c] offset = do
            let req = makeRequest offset True c
            sendFinalInput call (Proto.encodeWriteRequest req)
          sendChunks (c:cs) offset = do
            let req = makeRequest offset False c
            sendNextInput call (Proto.encodeWriteRequest req)
            sendChunks cs (offset + fromIntegral (BS.length c))
      sendChunks chunks 0
      (respBytes, _trailers) <- recvFinalOutput call
      pure respBytes

  case result of
    Left err ->
      putStrLn $ "CAS: stream upload failed: " <> show err
    Right respBytes ->
      case Proto.decodeWriteResponse respBytes of
        Nothing ->
          putStrLn $ "CAS: stream upload response decode failed"
        Just resp ->
          putStrLn $ "CAS: stream uploaded " <> show (Proto.wrCommittedSize resp) <> " bytes"

-- | Split bytestring into chunks
chunksOf :: Int -> ByteString -> [ByteString]
chunksOf n bs
  | BS.null bs = []
  | otherwise =
      let (chunk, rest) = BS.splitAt n bs
      in chunk : chunksOf n rest

-- | Download blob from CAS
--
-- Returns Nothing if blob not found.
downloadBlob :: CASClient -> Digest -> IO (Maybe ByteString)
downloadBlob client digest = do
  let resourceName = digestToResourceName (casInstanceName $ clientConfig client) digest
      request = Proto.ReadRequest
        { Proto.rrResourceName = resourceName
        , Proto.rrReadOffset = 0
        , Proto.rrReadLimit = 0  -- 0 means no limit
        }
      reqBytes = Proto.encodeReadRequest request

  result <- try @SomeException $
    withRPC (clientConn client) def (Proxy @Proto.ByteStreamRead) $ \call -> do
      sendFinalInput call reqBytes
      -- Collect all response chunks
      let collectChunks acc = do
            next <- recvNextOutputElem call
            case next of
              NoNextElem -> pure acc
              NextElem chunk ->
                case Proto.decodeReadResponse chunk of
                  Nothing -> collectChunks acc  -- skip bad chunk
                  Just resp -> collectChunks (acc <> Proto.rrData resp)
      collectChunks BS.empty

  case result of
    Left err -> do
      putStrLn $ "CAS: download failed: " <> show err
      pure Nothing
    Right content ->
      if BS.null content
        then pure Nothing
        else pure (Just content)

-- | Check which blobs are missing from CAS
--
-- Returns list of digests not present in the store.
findMissingBlobs :: CASClient -> [Digest] -> IO [Digest]
findMissingBlobs client digests = do
  let request = Proto.FindMissingBlobsRequest
        { Proto.fmbrInstanceName = casInstanceName (clientConfig client)
        , Proto.fmbrBlobDigests = map toProtoDigest digests
        }
      reqBytes = Proto.encodeFindMissingBlobsRequest request

  result <- try @SomeException $
    withRPC (clientConn client) def (Proxy @Proto.CASFindMissingBlobs) $ \call -> do
      sendFinalInput call reqBytes
      (respBytes, _trailers) <- recvFinalOutput call
      pure respBytes

  case result of
    Left err -> do
      putStrLn $ "CAS: findMissingBlobs failed: " <> show err
      -- On error, assume all are missing (safe fallback)
      pure digests
    Right respBytes ->
      case Proto.decodeFindMissingBlobsResponse respBytes of
        Nothing -> do
          putStrLn $ "CAS: findMissingBlobs response decode failed"
          pure digests
        Just resp ->
          pure $ map fromProtoDigest (Proto.fmbrMissingBlobDigests resp)

-- | Check if a single blob exists
blobExists :: CASClient -> Digest -> IO Bool
blobExists client digest = do
  missing <- findMissingBlobs client [digest]
  pure (null missing)
