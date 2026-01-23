{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}

{- | High-level interface to Nix values from WASM modules.

This module provides a safe(r) Haskell API for constructing and inspecting
Nix values during evaluation. Values are represented as opaque handles
managed by the host.

= Example usage

@
-- A simple function that doubles an integer
double :: Value -> IO Value
double v = do
  n <- getInt v
  mkInt (n * 2)

-- Export it for use from Nix
foreign export ccall "double" double :: Value -> IO Value
@
-}
module Aleph.Nix.Value (
    -- * The Value type
    Value (..),
    getType,

    -- * Panic and warnings
    panic,
    warn,

    -- * Constructing values
    mkInt,
    mkFloat,
    mkString,
    mkBool,
    mkNull,
    mkPath,
    mkList,
    mkAttrs,

    -- * Inspecting values
    getInt,
    getFloat,
    getString,
    getStringLen,
    getBool,
    getPath,
    getList,
    getListLen,
    getListElem,
    getAttrs,
    getAttrsLen,
    getAttr,
    hasAttr,
    lookupAttr,

    -- * Calling Nix functions
    call,
    call1,
    call2,

    -- * Module initialization
    nixWasmInit,
) where

import qualified Data.ByteString as BS
import qualified Data.ByteString.Unsafe as BS
import Data.Int (Int64)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import Data.Word (Word32, Word8)
import Foreign.C.Types (CDouble (..), CSize (..))
import Foreign.Marshal.Alloc (allocaBytes)
import Foreign.Marshal.Array (allocaArray, peekArray, pokeArray)
import Foreign.Ptr (Ptr, castPtr, ptrToWordPtr)
import Foreign.Storable (Storable (..), peekByteOff, pokeByteOff)

import Aleph.Nix.FFI (
    ValueId,
    c_call_function,
    c_copy_attrname,
    c_copy_attrset,
    c_copy_list,
    c_copy_path,
    c_copy_string,
    c_get_attr,
    c_get_attrs_len,
    c_get_bool,
    c_get_float,
    c_get_int,
    c_get_list_elem,
    c_get_list_len,
    c_get_string_len,
    c_get_type,
    c_has_attr,
    c_make_attrset,
    c_make_bool,
    c_make_float,
    c_make_int,
    c_make_list,
    c_make_null,
    c_make_path,
    c_make_string,
    c_panic,
    c_warn,
 )
import Aleph.Nix.Types

{- | A handle to a Nix value in the host evaluator.

Values are opaque references managed by the host. They are valid for the
duration of the WASM function call but should not be persisted across calls.
-}
newtype Value = Value {unValue :: ValueId}
    deriving newtype (Eq, Show, Storable)

--------------------------------------------------------------------------------
-- Type inspection
--------------------------------------------------------------------------------

-- | Get the type of a Nix value.
getType :: Value -> IO NixType
getType (Value v) = fromTypeId <$> c_get_type v

--------------------------------------------------------------------------------
-- Panic and warnings
--------------------------------------------------------------------------------

{- | Abort evaluation with an error message.
This function does not return.
-}
panic :: Text -> IO a
panic msg = do
    let bs = T.encodeUtf8 msg
    BS.unsafeUseAsCStringLen bs $ \(ptr, len) ->
        c_panic (castPtr ptr) (fromIntegral len)
    -- The host should abort, but just in case:
    error "panic did not abort"

-- | Emit a warning during evaluation.
warn :: Text -> IO ()
warn msg = do
    let bs = T.encodeUtf8 msg
    BS.unsafeUseAsCStringLen bs $ \(ptr, len) ->
        c_warn (castPtr ptr) (fromIntegral len)

--------------------------------------------------------------------------------
-- Constructing values
--------------------------------------------------------------------------------

-- | Create a Nix integer.
mkInt :: Int64 -> IO Value
mkInt n = Value <$> c_make_int n

-- | Create a Nix float.
mkFloat :: Double -> IO Value
mkFloat f = Value <$> c_make_float (CDouble f)

-- | Create a Nix string from Text.
mkString :: Text -> IO Value
mkString t = do
    let bs = T.encodeUtf8 t
    BS.unsafeUseAsCStringLen bs $ \(ptr, len) ->
        Value <$> c_make_string (castPtr ptr) (fromIntegral len)

-- | Create a Nix boolean.
mkBool :: Bool -> IO Value
mkBool b = Value <$> c_make_bool (if b then 1 else 0)

-- | Create the Nix null value.
mkNull :: IO Value
mkNull = Value <$> c_make_null

-- | Create a Nix path from Text.
mkPath :: Text -> IO Value
mkPath t = do
    let bs = T.encodeUtf8 t
    BS.unsafeUseAsCStringLen bs $ \(ptr, len) ->
        Value <$> c_make_path (castPtr ptr) (fromIntegral len)

-- | Create a Nix list from a list of values.
mkList :: [Value] -> IO Value
mkList vs = allocaArray (length vs) $ \ptr -> do
    pokeArray ptr vs
    Value <$> c_make_list (castPtr ptr) (fromIntegral $ length vs)

{- | Create a Nix attribute set from a Map.

The attrset layout expected by the host is an array of entries where each
entry is: (name_ptr: u32, name_len: u32, value: u32).
-}
mkAttrs :: Map Text Value -> IO Value
mkAttrs attrs = do
    let entries = Map.toList attrs
        nAttrs = length entries
        -- Each entry: ptr (4) + len (4) + value (4) = 12 bytes
        entrySize = 12
    allocaBytes (nAttrs * entrySize) $ \basePtr -> do
        -- We need to keep the ByteStrings alive, so we process them one by one
        -- and poke into the buffer. This is a bit inefficient but safe.
        let pokeEntry :: Int -> (Text, Value) -> IO ()
            pokeEntry idx (name, Value val) = do
                let bs = T.encodeUtf8 name
                    offset = idx * entrySize
                BS.unsafeUseAsCStringLen bs $ \(namePtr, nameLen) -> do
                    -- NOTE: This is tricky - we're storing the pointer which may become
                    -- invalid. The host must copy the string immediately.
                    -- In WASM32, pointers are 32-bit.
                    pokeByteOff basePtr offset (fromIntegral (ptrToWord32 namePtr) :: Word32)
                    pokeByteOff basePtr (offset + 4) (fromIntegral nameLen :: Word32)
                    pokeByteOff basePtr (offset + 8) val
        mapM_ (uncurry pokeEntry) (zip [0 ..] entries)
        Value <$> c_make_attrset basePtr (fromIntegral nAttrs)
  where
    -- Convert a pointer to Word32 (valid for WASM32 where pointers are 32-bit)
    ptrToWord32 :: Ptr a -> Word32
    ptrToWord32 = fromIntegral . ptrToWordPtr

--------------------------------------------------------------------------------
-- Inspecting values
--------------------------------------------------------------------------------

-- | Extract an integer from a Nix value.
getInt :: Value -> IO Int64
getInt (Value v) = c_get_int v

-- | Extract a float from a Nix value.
getFloat :: Value -> IO Double
getFloat (Value v) = do
    CDouble f <- c_get_float v
    pure f

-- | Extract a string from a Nix value as Text.
getString :: Value -> IO Text
getString (Value v) = do
    -- First try with a small stack buffer
    let smallBufSize = 256
    allocaBytes smallBufSize $ \buf -> do
        actualLen <- c_copy_string v buf (fromIntegral smallBufSize)
        if fromIntegral actualLen <= smallBufSize
            then do
                bs <- BS.packCStringLen (castPtr buf, fromIntegral actualLen)
                pure (T.decodeUtf8 bs)
            else do
                -- Need a bigger buffer
                allocaBytes (fromIntegral actualLen) $ \bigBuf -> do
                    _ <- c_copy_string v bigBuf actualLen
                    bs <- BS.packCStringLen (castPtr bigBuf, fromIntegral actualLen)
                    pure (T.decodeUtf8 bs)

-- | Get the length of a string without copying.
getStringLen :: Value -> IO Int
getStringLen (Value v) = fromIntegral <$> c_get_string_len v

-- | Extract a boolean from a Nix value.
getBool :: Value -> IO Bool
getBool (Value v) = do
    b <- c_get_bool v
    pure (b /= 0)

-- | Extract a path from a Nix value as Text.
getPath :: Value -> IO Text
getPath (Value v) = do
    let smallBufSize = 256
    allocaBytes smallBufSize $ \buf -> do
        actualLen <- c_copy_path v buf (fromIntegral smallBufSize)
        if fromIntegral actualLen <= smallBufSize
            then do
                bs <- BS.packCStringLen (castPtr buf, fromIntegral actualLen)
                pure (T.decodeUtf8 bs)
            else do
                allocaBytes (fromIntegral actualLen) $ \bigBuf -> do
                    _ <- c_copy_path v bigBuf actualLen
                    bs <- BS.packCStringLen (castPtr bigBuf, fromIntegral actualLen)
                    pure (T.decodeUtf8 bs)

-- | Extract a list from a Nix value.
getList :: Value -> IO [Value]
getList (Value v) = do
    -- First try with a small stack buffer
    let smallBufSize = 64
    allocaArray smallBufSize $ \buf -> do
        actualLen <- c_copy_list v (castPtr buf) (fromIntegral smallBufSize)
        if fromIntegral actualLen <= smallBufSize
            then map Value <$> peekArray (fromIntegral actualLen) buf
            else do
                -- Need a bigger buffer
                allocaArray (fromIntegral actualLen) $ \bigBuf -> do
                    _ <- c_copy_list v (castPtr bigBuf) actualLen
                    map Value <$> peekArray (fromIntegral actualLen) bigBuf

-- | Get the length of a list without copying.
getListLen :: Value -> IO Int
getListLen (Value v) = fromIntegral <$> c_get_list_len v

-- | Get a single list element by index.
getListElem :: Value -> Int -> IO Value
getListElem (Value v) idx = Value <$> c_get_list_elem v (fromIntegral idx)

{- | Extract an attribute set from a Nix value.

The host returns an array of (value_id, name_len) pairs, then we fetch
each name separately with copy_attrname.
-}
getAttrs :: Value -> IO (Map Text Value)
getAttrs (Value v) = do
    -- First get the entries (value, name_len pairs)
    let smallBufSize = 32 -- number of attrs
        entrySize = 8 -- value_id (4) + name_len (4)
    allocaBytes (smallBufSize * entrySize) $ \buf -> do
        actualLen <- c_copy_attrset v buf (fromIntegral smallBufSize)
        entries <-
            if fromIntegral actualLen <= smallBufSize
                then peekEntries buf (fromIntegral actualLen)
                else allocaBytes (fromIntegral actualLen * entrySize) $ \bigBuf -> do
                    _ <- c_copy_attrset v bigBuf actualLen
                    peekEntries bigBuf (fromIntegral actualLen)
        -- Now fetch each attribute name
        fmap Map.fromList $ forM (zip [0 ..] entries) $ \(idx, (valId, nameLen)) -> do
            name <- allocaBytes (fromIntegral nameLen) $ \nameBuf -> do
                c_copy_attrname v (fromIntegral idx) nameBuf (fromIntegral nameLen)
                bs <- BS.packCStringLen (castPtr nameBuf, fromIntegral nameLen)
                pure (T.decodeUtf8 bs)
            pure (name, Value valId)
  where
    peekEntries :: Ptr Word8 -> Int -> IO [(ValueId, Word32)]
    peekEntries ptr n = forM [0 .. n - 1] $ \i -> do
        let offset = i * 8
        valId <- peekByteOff ptr offset :: IO Word32
        nameLen <- peekByteOff ptr (offset + 4) :: IO Word32
        pure (valId, nameLen)

    forM = flip mapM

-- | Get the number of attributes without copying.
getAttrsLen :: Value -> IO Int
getAttrsLen (Value v) = fromIntegral <$> c_get_attrs_len v

-- | Check if an attribute exists.
hasAttr :: Value -> Text -> IO Bool
hasAttr (Value v) name = do
    let bs = T.encodeUtf8 name
    BS.unsafeUseAsCStringLen bs $ \(ptr, len) -> do
        result <- c_has_attr v (castPtr ptr) (fromIntegral len)
        pure (result /= 0)

-- | Get an attribute by name. Returns Nothing if not found.
getAttr :: Value -> Text -> IO (Maybe Value)
getAttr (Value v) name = do
    let bs = T.encodeUtf8 name
    BS.unsafeUseAsCStringLen bs $ \(ptr, len) -> do
        result <- c_get_attr v (castPtr ptr) (fromIntegral len)
        if result == 0xFFFFFFFF
            then pure Nothing
            else pure (Just (Value result))

-- | Get an attribute by name, panicking if not found.
lookupAttr :: Value -> Text -> IO Value
lookupAttr v name = do
    mval <- getAttr v name
    case mval of
        Just val -> pure val
        Nothing -> panic $ T.concat ["attribute '", name, "' not found"]

--------------------------------------------------------------------------------
-- Calling Nix functions
--------------------------------------------------------------------------------

-- | Call a Nix function with a list of arguments.
call :: Value -> [Value] -> IO Value
call (Value fun) args = allocaArray (length args) $ \ptr -> do
    pokeArray ptr args
    Value <$> c_call_function fun (castPtr ptr) (fromIntegral $ length args)

-- | Call a Nix function with one argument.
call1 :: Value -> Value -> IO Value
call1 fun arg = call fun [arg]

-- | Call a Nix function with two arguments.
call2 :: Value -> Value -> Value -> IO Value
call2 fun arg1 arg2 = call fun [arg1, arg2]

--------------------------------------------------------------------------------
-- Module initialization
--------------------------------------------------------------------------------

{- | Initialize the WASM module.

This should be called from 'nix_wasm_init_v1'. It sets up exception handling
so that Haskell exceptions are converted to Nix panics.

Usage:

@
foreign export ccall "nix_wasm_init_v1" init :: IO ()

init :: IO ()
init = nixWasmInit
@
-}
nixWasmInit :: IO ()
nixWasmInit = pure ()

-- In a full implementation, we might set up exception handlers here.
-- For now, it's a no-op since GHC's RTS handles most things.
