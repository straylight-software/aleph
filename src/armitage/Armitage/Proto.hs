{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

{- |
Module      : Armitage.Proto
Description : Hand-written protobuf types for Remote Execution API

Manual protobuf encoding for REAPI types. This avoids the proto-lens
build dependency while still producing wire-compatible messages.

Protobuf wire format reference:
  https://protobuf.dev/programming-guides/encoding/

Field types used:
  - string: wire type 2 (length-delimited), UTF-8 encoded
  - bytes:  wire type 2 (length-delimited)
  - int64:  wire type 0 (varint)
  - bool:   wire type 0 (varint, 0 or 1)
  - message: wire type 2 (length-delimited)
  - repeated: multiple fields with same tag

Key = (field_number << 3) | wire_type

Uses grapesy's RawRpc for gRPC without proto-lens dependency.
Input/Output are lazy ByteStrings that we encode/decode manually.
-}
module Armitage.Proto
  ( -- * Wire encoding primitives
    encodeVarint
  , decodeVarint
  , encodeField
  , encodeString
  , encodeBytes
  , encodeMessage

    -- * Digest (used everywhere)
  , ProtoDigest (..)
  , encodeDigest
  , decodeDigest

    -- * CAS: FindMissingBlobs
  , FindMissingBlobsRequest (..)
  , FindMissingBlobsResponse (..)
  , encodeFindMissingBlobsRequest
  , decodeFindMissingBlobsResponse

    -- * CAS: BatchUpdateBlobs
  , BatchUpdateBlobsRequest (..)
  , BatchUpdateBlobsResponse (..)
  , BlobRequest (..)
  , BlobResponse (..)
  , encodeBatchUpdateBlobsRequest
  , decodeBatchUpdateBlobsResponse

    -- * CAS: BatchReadBlobs
  , BatchReadBlobsRequest (..)
  , BatchReadBlobsResponse (..)
  , ReadBlobResponse (..)
  , encodeBatchReadBlobsRequest
  , decodeBatchReadBlobsResponse

    -- * ByteStream: Read
  , ReadRequest (..)
  , ReadResponse (..)
  , encodeReadRequest
  , decodeReadResponse

    -- * ByteStream: Write
  , WriteRequest (..)
  , WriteResponse (..)
  , encodeWriteRequest
  , decodeWriteResponse

    -- * gRPC RPC type aliases
  , CASFindMissingBlobs
  , CASBatchUpdateBlobs
  , CASBatchReadBlobs
  , ByteStreamRead
  , ByteStreamWrite
  ) where

import Data.Bits (shiftL, shiftR, (.&.), (.|.))
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import Data.Int (Int64)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Word (Word64)
import GHC.Generics (Generic)

-- grapesy RawRpc for defining RPCs without proto-lens
import Network.GRPC.Spec
  ( RawRpc
  , RequestMetadata
  , ResponseInitialMetadata
  , ResponseTrailingMetadata
  )
import Network.GRPC.Common (NoMetadata)

-- =============================================================================
-- RPC Type Aliases
-- =============================================================================

-- | FindMissingBlobs: Non-streaming RPC
type CASFindMissingBlobs = RawRpc
  "build.bazel.remote.execution.v2.ContentAddressableStorage"
  "FindMissingBlobs"

-- | BatchUpdateBlobs: Non-streaming RPC
type CASBatchUpdateBlobs = RawRpc
  "build.bazel.remote.execution.v2.ContentAddressableStorage"
  "BatchUpdateBlobs"

-- | BatchReadBlobs: Non-streaming RPC
type CASBatchReadBlobs = RawRpc
  "build.bazel.remote.execution.v2.ContentAddressableStorage"
  "BatchReadBlobs"

-- | ByteStream.Read: Server-streaming RPC
type ByteStreamRead = RawRpc
  "google.bytestream.ByteStream"
  "Read"

-- | ByteStream.Write: Client-streaming RPC
type ByteStreamWrite = RawRpc
  "google.bytestream.ByteStream"
  "Write"

-- =============================================================================
-- Wire format primitives
-- =============================================================================

-- | Encode unsigned varint (LEB128)
encodeVarint :: Word64 -> ByteString
encodeVarint n
  | n < 128   = BS.singleton (fromIntegral n)
  | otherwise = BS.cons (fromIntegral (n .&. 0x7F) .|. 0x80)
                        (encodeVarint (n `shiftR` 7))

-- | Encode signed varint (for int64)
encodeSignedVarint :: Int64 -> ByteString
encodeSignedVarint n = encodeVarint (fromIntegral n :: Word64)

-- | Decode varint from bytestring, return (value, remaining bytes)
decodeVarint :: ByteString -> Maybe (Word64, ByteString)
decodeVarint bs = go bs 0 0
  where
    go remaining acc shift
      | BS.null remaining = Nothing
      | otherwise =
          let byte = BS.head remaining
              rest = BS.tail remaining
              val  = fromIntegral (byte .&. 0x7F) `shiftL` shift
              acc' = acc .|. val
          in if byte .&. 0x80 == 0
             then Just (acc', rest)
             else if shift >= 63
                  then Nothing  -- overflow protection
                  else go rest acc' (shift + 7)

-- | Wire types
wireVarint, wireLengthDelimited :: Word64
wireVarint = 0
wireLengthDelimited = 2

-- | Encode field key (tag = field_number << 3 | wire_type)
encodeFieldKey :: Int -> Word64 -> ByteString
encodeFieldKey fieldNum wireType =
  encodeVarint (fromIntegral fieldNum `shiftL` 3 .|. wireType)

-- | Encode a length-delimited field (string, bytes, message)
encodeField :: Int -> ByteString -> ByteString
encodeField fieldNum content =
  encodeFieldKey fieldNum wireLengthDelimited
  <> encodeVarint (fromIntegral $ BS.length content)
  <> content

-- | Encode a varint field
encodeVarintField :: Int -> Word64 -> ByteString
encodeVarintField fieldNum val =
  encodeFieldKey fieldNum wireVarint <> encodeVarint val

-- | Encode a signed int64 field
encodeInt64Field :: Int -> Int64 -> ByteString
encodeInt64Field fieldNum val =
  encodeFieldKey fieldNum wireVarint <> encodeSignedVarint val

-- | Encode a bool field
encodeBoolField :: Int -> Bool -> ByteString
encodeBoolField fieldNum val =
  encodeVarintField fieldNum (if val then 1 else 0)

-- | Encode string field (UTF-8)
encodeString :: Int -> Text -> ByteString
encodeString fieldNum txt =
  encodeField fieldNum (TE.encodeUtf8 txt)

-- | Encode bytes field
encodeBytes :: Int -> ByteString -> ByteString
encodeBytes = encodeField

-- | Encode embedded message
encodeMessage :: Int -> ByteString -> ByteString
encodeMessage = encodeField

-- =============================================================================
-- Decoding helpers
-- =============================================================================

-- | Parse a single field from wire format
-- Returns (field_number, wire_type, field_data, remaining)
parseField :: ByteString -> Maybe (Int, Word64, ByteString, ByteString)
parseField bs = do
  (key, rest1) <- decodeVarint bs
  let fieldNum = fromIntegral (key `shiftR` 3)
      wireType = key .&. 0x07
  case wireType of
    0 -> do  -- varint
      (val, rest2) <- decodeVarint rest1
      Just (fieldNum, wireType, encodeVarint val, rest2)
    2 -> do  -- length-delimited
      (len, rest2) <- decodeVarint rest1
      let (content, rest3) = BS.splitAt (fromIntegral len) rest2
      Just (fieldNum, wireType, content, rest3)
    _ -> Nothing  -- unsupported wire type for our use case

-- | Parse all fields into a list
parseAllFields :: ByteString -> [(Int, ByteString)]
parseAllFields bs
  | BS.null bs = []
  | otherwise = case parseField bs of
      Nothing -> []
      Just (fieldNum, _, content, rest) ->
        (fieldNum, content) : parseAllFields rest

-- | Get field by number (returns first match)
getField :: Int -> [(Int, ByteString)] -> Maybe ByteString
getField n fields = lookup n fields

-- | Get all fields with given number (for repeated)
getRepeatedFields :: Int -> [(Int, ByteString)] -> [ByteString]
getRepeatedFields n fields = [content | (fn, content) <- fields, fn == n]

-- | Decode text from field content
decodeText :: ByteString -> Text
decodeText = TE.decodeUtf8

-- | Decode int64 from varint bytes
decodeInt64 :: ByteString -> Int64
decodeInt64 bs = case decodeVarint bs of
  Just (v, _) -> fromIntegral v
  Nothing -> 0

-- =============================================================================
-- Digest
-- =============================================================================

-- | Protobuf Digest message
--
-- message Digest {
--   string hash = 1;
--   int64 size_bytes = 2;
-- }
data ProtoDigest = ProtoDigest
  { pdHash :: Text       -- ^ SHA256 hex (64 chars)
  , pdSizeBytes :: Int64 -- ^ Size in bytes
  }
  deriving (Show, Eq, Generic)

encodeDigest :: ProtoDigest -> ByteString
encodeDigest ProtoDigest{..} =
  encodeString 1 pdHash
  <> encodeInt64Field 2 pdSizeBytes

decodeDigest :: ByteString -> Maybe ProtoDigest
decodeDigest bs = do
  let fields = parseAllFields bs
  hashField <- getField 1 fields
  sizeField <- getField 2 fields
  Just ProtoDigest
    { pdHash = decodeText hashField
    , pdSizeBytes = decodeInt64 sizeField
    }

-- =============================================================================
-- FindMissingBlobs
-- =============================================================================

-- | Request for FindMissingBlobs RPC
--
-- message FindMissingBlobsRequest {
--   string instance_name = 1;
--   repeated Digest blob_digests = 2;
-- }
data FindMissingBlobsRequest = FindMissingBlobsRequest
  { fmbrInstanceName :: Text
  , fmbrBlobDigests :: [ProtoDigest]
  }
  deriving (Show, Eq, Generic)

-- | Response for FindMissingBlobs RPC
--
-- message FindMissingBlobsResponse {
--   repeated Digest missing_blob_digests = 2;
-- }
data FindMissingBlobsResponse = FindMissingBlobsResponse
  { fmbrMissingBlobDigests :: [ProtoDigest]
  }
  deriving (Show, Eq, Generic)

encodeFindMissingBlobsRequest :: FindMissingBlobsRequest -> LBS.ByteString
encodeFindMissingBlobsRequest FindMissingBlobsRequest{..} =
  LBS.fromStrict $
    encodeString 1 fmbrInstanceName
    <> mconcat [encodeMessage 2 (encodeDigest d) | d <- fmbrBlobDigests]

decodeFindMissingBlobsResponse :: LBS.ByteString -> Maybe FindMissingBlobsResponse
decodeFindMissingBlobsResponse lbs = do
  let bs = LBS.toStrict lbs
      fields = parseAllFields bs
      digestFields = getRepeatedFields 2 fields
  digests <- mapM decodeDigest digestFields
  Just FindMissingBlobsResponse { fmbrMissingBlobDigests = digests }

-- =============================================================================
-- BatchUpdateBlobs
-- =============================================================================

-- | Single blob upload request
data BlobRequest = BlobRequest
  { brDigest :: ProtoDigest
  , brData :: ByteString
  }
  deriving (Show, Eq, Generic)

-- | Request for BatchUpdateBlobs RPC
data BatchUpdateBlobsRequest = BatchUpdateBlobsRequest
  { bubrInstanceName :: Text
  , bubrRequests :: [BlobRequest]
  }
  deriving (Show, Eq, Generic)

-- | Single blob upload response
data BlobResponse = BlobResponse
  { brespDigest :: ProtoDigest
  , brespStatusCode :: Int  -- google.rpc.Status code (0 = OK)
  }
  deriving (Show, Eq, Generic)

-- | Response for BatchUpdateBlobs RPC
data BatchUpdateBlobsResponse = BatchUpdateBlobsResponse
  { bubrResponses :: [BlobResponse]
  }
  deriving (Show, Eq, Generic)

encodeBlobRequest :: BlobRequest -> ByteString
encodeBlobRequest BlobRequest{..} =
  encodeMessage 1 (encodeDigest brDigest)
  <> encodeBytes 2 brData

encodeBatchUpdateBlobsRequest :: BatchUpdateBlobsRequest -> LBS.ByteString
encodeBatchUpdateBlobsRequest BatchUpdateBlobsRequest{..} =
  LBS.fromStrict $
    encodeString 1 bubrInstanceName
    <> mconcat [encodeMessage 2 (encodeBlobRequest r) | r <- bubrRequests]

decodeBlobResponse :: ByteString -> Maybe BlobResponse
decodeBlobResponse bs = do
  let fields = parseAllFields bs
  digestField <- getField 1 fields
  digest <- decodeDigest digestField
  -- Status is a nested message, field 2; code is field 1 within that
  let statusCode = case getField 2 fields of
        Nothing -> 0
        Just statusBs ->
          let statusFields = parseAllFields statusBs
          in case getField 1 statusFields of
               Nothing -> 0
               Just codeBs -> fromIntegral $ decodeInt64 codeBs
  Just BlobResponse
    { brespDigest = digest
    , brespStatusCode = statusCode
    }

decodeBatchUpdateBlobsResponse :: LBS.ByteString -> Maybe BatchUpdateBlobsResponse
decodeBatchUpdateBlobsResponse lbs = do
  let bs = LBS.toStrict lbs
      fields = parseAllFields bs
      respFields = getRepeatedFields 1 fields
  resps <- mapM decodeBlobResponse respFields
  Just BatchUpdateBlobsResponse { bubrResponses = resps }

-- =============================================================================
-- BatchReadBlobs
-- =============================================================================

-- | Request for BatchReadBlobs RPC
data BatchReadBlobsRequest = BatchReadBlobsRequest
  { brbrInstanceName :: Text
  , brbrDigests :: [ProtoDigest]
  }
  deriving (Show, Eq, Generic)

-- | Single blob read response
data ReadBlobResponse = ReadBlobResponse
  { rbrDigest :: ProtoDigest
  , rbrData :: ByteString
  , rbrStatusCode :: Int
  }
  deriving (Show, Eq, Generic)

-- | Response for BatchReadBlobs RPC
data BatchReadBlobsResponse = BatchReadBlobsResponse
  { brbrResponses :: [ReadBlobResponse]
  }
  deriving (Show, Eq, Generic)

encodeBatchReadBlobsRequest :: BatchReadBlobsRequest -> LBS.ByteString
encodeBatchReadBlobsRequest BatchReadBlobsRequest{..} =
  LBS.fromStrict $
    encodeString 1 brbrInstanceName
    <> mconcat [encodeMessage 2 (encodeDigest d) | d <- brbrDigests]

decodeReadBlobResponse :: ByteString -> Maybe ReadBlobResponse
decodeReadBlobResponse bs = do
  let fields = parseAllFields bs
  digestField <- getField 1 fields
  digest <- decodeDigest digestField
  let blobData = maybe BS.empty id (getField 2 fields)
      statusCode = case getField 3 fields of
        Nothing -> 0
        Just statusBs ->
          let statusFields = parseAllFields statusBs
          in case getField 1 statusFields of
               Nothing -> 0
               Just codeBs -> fromIntegral $ decodeInt64 codeBs
  Just ReadBlobResponse
    { rbrDigest = digest
    , rbrData = blobData
    , rbrStatusCode = statusCode
    }

decodeBatchReadBlobsResponse :: LBS.ByteString -> Maybe BatchReadBlobsResponse
decodeBatchReadBlobsResponse lbs = do
  let bs = LBS.toStrict lbs
      fields = parseAllFields bs
      respFields = getRepeatedFields 1 fields
  resps <- mapM decodeReadBlobResponse respFields
  Just BatchReadBlobsResponse { brbrResponses = resps }

-- =============================================================================
-- ByteStream Read
-- =============================================================================

-- | Request for ByteStream.Read
data ReadRequest = ReadRequest
  { rrResourceName :: Text
  , rrReadOffset :: Int64
  , rrReadLimit :: Int64
  }
  deriving (Show, Eq, Generic)

-- | Response chunk for ByteStream.Read
data ReadResponse = ReadResponse
  { rrData :: ByteString
  }
  deriving (Show, Eq, Generic)

encodeReadRequest :: ReadRequest -> LBS.ByteString
encodeReadRequest ReadRequest{..} =
  LBS.fromStrict $
    encodeString 1 rrResourceName
    <> (if rrReadOffset /= 0 then encodeInt64Field 2 rrReadOffset else mempty)
    <> (if rrReadLimit /= 0 then encodeInt64Field 3 rrReadLimit else mempty)

decodeReadResponse :: LBS.ByteString -> Maybe ReadResponse
decodeReadResponse lbs = do
  let bs = LBS.toStrict lbs
      fields = parseAllFields bs
  -- data field is 10 in the proto
  let blobData = maybe BS.empty id (getField 10 fields)
  Just ReadResponse { rrData = blobData }

-- =============================================================================
-- ByteStream Write
-- =============================================================================

-- | Request chunk for ByteStream.Write
data WriteRequest = WriteRequest
  { wrResourceName :: Text
  , wrWriteOffset :: Int64
  , wrFinishWrite :: Bool
  , wrData :: ByteString
  }
  deriving (Show, Eq, Generic)

-- | Response for ByteStream.Write
data WriteResponse = WriteResponse
  { wrCommittedSize :: Int64
  }
  deriving (Show, Eq, Generic)

encodeWriteRequest :: WriteRequest -> LBS.ByteString
encodeWriteRequest WriteRequest{..} =
  LBS.fromStrict $
    encodeString 1 wrResourceName
    <> (if wrWriteOffset /= 0 then encodeInt64Field 2 wrWriteOffset else mempty)
    <> (if wrFinishWrite then encodeBoolField 3 True else mempty)
    <> (if not (BS.null wrData) then encodeBytes 10 wrData else mempty)

decodeWriteResponse :: LBS.ByteString -> Maybe WriteResponse
decodeWriteResponse lbs = do
  let bs = LBS.toStrict lbs
      fields = parseAllFields bs
  let committed = case getField 1 fields of
        Nothing -> 0
        Just f -> decodeInt64 f
  Just WriteResponse { wrCommittedSize = committed }

-- =============================================================================
-- gRPC Metadata type instances
-- =============================================================================
-- RawRpc needs these type family instances to work with grapesy

-- CASFindMissingBlobs
type instance RequestMetadata CASFindMissingBlobs = NoMetadata
type instance ResponseInitialMetadata CASFindMissingBlobs = NoMetadata
type instance ResponseTrailingMetadata CASFindMissingBlobs = NoMetadata

-- CASBatchUpdateBlobs
type instance RequestMetadata CASBatchUpdateBlobs = NoMetadata
type instance ResponseInitialMetadata CASBatchUpdateBlobs = NoMetadata
type instance ResponseTrailingMetadata CASBatchUpdateBlobs = NoMetadata

-- CASBatchReadBlobs
type instance RequestMetadata CASBatchReadBlobs = NoMetadata
type instance ResponseInitialMetadata CASBatchReadBlobs = NoMetadata
type instance ResponseTrailingMetadata CASBatchReadBlobs = NoMetadata

-- ByteStreamRead
type instance RequestMetadata ByteStreamRead = NoMetadata
type instance ResponseInitialMetadata ByteStreamRead = NoMetadata
type instance ResponseTrailingMetadata ByteStreamRead = NoMetadata

-- ByteStreamWrite
type instance RequestMetadata ByteStreamWrite = NoMetadata
type instance ResponseInitialMetadata ByteStreamWrite = NoMetadata
type instance ResponseTrailingMetadata ByteStreamWrite = NoMetadata
