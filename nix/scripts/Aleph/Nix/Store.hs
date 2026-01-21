{-# LANGUAGE CApiFFI #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE OverloadedStrings #-}

{- | Store primitives - call INTO Nix to resolve dependencies.

This inverts the control flow: Haskell calls Nix for dep resolution,
rather than Nix passing deps as env vars. The WASI host exposes these
primitives, and Haskell orchestrates the build.

= FFI to straylight-nix host

The host provides:
  - resolve_dep: Resolve a package name to its store path
  - add_to_store: Add a path to the store (for outputs)
  - get_system: Get the current system string

= Usage

@
import Aleph.Nix.Store

main = do
  -- Resolve cmake dependency
  cmakePath <- resolveDep "cmake"
  
  -- cmakePath is now /nix/store/xyz-cmake-3.28.0
  let cmakeBin = cmakePath <> "/bin/cmake"
@
-}
module Aleph.Nix.Store (
    -- * Dependency resolution
    resolveDep,
    resolveDeps,
    
    -- * Store operations
    addToStore,
    getStorePath,
    
    -- * System info
    getSystem,
    getCores,
    getOutPath,
    
    -- * Low-level FFI
    c_resolve_dep,
    c_add_to_store,
    c_get_system,
    c_get_cores,
    c_get_out_path,
) where

import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.ByteString as BS
import qualified Data.ByteString.Unsafe as BS
import Data.Word (Word8, Word32)
import Foreign.C.Types (CSize (..))
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Marshal.Alloc (allocaBytes)

--------------------------------------------------------------------------------
-- Low-level FFI
--------------------------------------------------------------------------------

-- | Resolve a dependency name to its store path.
-- Parameters: name, name_len, output_buf, output_buf_len
-- Returns: actual length of store path (0 on failure)
foreign import ccall unsafe "nix_resolve_dep"
    c_resolve_dep :: Ptr Word8 -> CSize   -- name
                  -> Ptr Word8 -> CSize   -- output buffer
                  -> IO CSize             -- actual length

-- | Add a path to the store.
-- Parameters: path, path_len, output_buf, output_buf_len
-- Returns: actual length of store path
foreign import ccall unsafe "nix_add_to_store"
    c_add_to_store :: Ptr Word8 -> CSize  -- path to add
                   -> Ptr Word8 -> CSize  -- output buffer
                   -> IO CSize            -- actual length

-- | Get the current system string.
-- Parameters: output_buf, output_buf_len
-- Returns: actual length
foreign import ccall unsafe "nix_get_system"
    c_get_system :: Ptr Word8 -> CSize -> IO CSize

-- | Get the number of CPU cores available.
foreign import ccall unsafe "nix_get_cores"
    c_get_cores :: IO Word32

-- | Get the output path ($out).
-- Parameters: output_name, name_len, output_buf, output_buf_len
-- Returns: actual length
foreign import ccall unsafe "nix_get_out_path"
    c_get_out_path :: Ptr Word8 -> CSize  -- output name (e.g., "out", "dev")
                   -> Ptr Word8 -> CSize  -- output buffer
                   -> IO CSize            -- actual length

--------------------------------------------------------------------------------
-- High-level API
--------------------------------------------------------------------------------

-- | Resolve a dependency by name to its store path.
--
-- @
-- cmakePath <- resolveDep "cmake"
-- -- Returns: "/nix/store/abc123-cmake-3.28.0"
-- @
resolveDep :: Text    -- ^ Package name
           -> IO Text -- ^ Store path
resolveDep name = do
    let nameBs = T.encodeUtf8 name
        bufSize = 256
    
    allocaBytes bufSize $ \outBuf ->
        BS.unsafeUseAsCStringLen nameBs $ \(namePtr, nameLen) -> do
            actualLen <- c_resolve_dep
                (castPtr namePtr) (fromIntegral nameLen)
                outBuf (fromIntegral bufSize)
            
            if actualLen == 0
                then error $ "resolveDep failed for: " <> T.unpack name
                else do
                    bs <- BS.packCStringLen (castPtr outBuf, fromIntegral actualLen)
                    pure (T.decodeUtf8 bs)

-- | Resolve multiple dependencies.
resolveDeps :: [Text] -> IO [(Text, Text)]
resolveDeps names = mapM (\n -> (n,) <$> resolveDep n) names

-- | Add a path to the Nix store.
--
-- This is used to add build outputs to the store.
addToStore :: Text    -- ^ Path to add
           -> IO Text -- ^ Store path
addToStore path = do
    let pathBs = T.encodeUtf8 path
        bufSize = 256
    
    allocaBytes bufSize $ \outBuf ->
        BS.unsafeUseAsCStringLen pathBs $ \(pathPtr, pathLen) -> do
            actualLen <- c_add_to_store
                (castPtr pathPtr) (fromIntegral pathLen)
                outBuf (fromIntegral bufSize)
            
            if actualLen == 0
                then error $ "addToStore failed for: " <> T.unpack path
                else do
                    bs <- BS.packCStringLen (castPtr outBuf, fromIntegral actualLen)
                    pure (T.decodeUtf8 bs)

-- | Get the store path for a given store path (identity, but validates).
getStorePath :: Text -> IO Text
getStorePath = pure  -- TODO: validate it's actually a store path

-- | Get the current system string (e.g., "x86_64-linux").
getSystem :: IO Text
getSystem = do
    let bufSize = 64
    allocaBytes bufSize $ \outBuf -> do
        actualLen <- c_get_system outBuf (fromIntegral bufSize)
        if actualLen == 0
            then pure "x86_64-linux"  -- fallback
            else do
                bs <- BS.packCStringLen (castPtr outBuf, fromIntegral actualLen)
                pure (T.decodeUtf8 bs)

-- | Get the number of CPU cores available.
getCores :: IO Int
getCores = fromIntegral <$> c_get_cores

-- | Get the output path for a named output.
--
-- @
-- outPath <- getOutPath "out"  -- main output
-- devPath <- getOutPath "dev"  -- dev output (if exists)
-- @
getOutPath :: Text    -- ^ Output name
           -> IO Text -- ^ Store path
getOutPath name = do
    let nameBs = T.encodeUtf8 name
        bufSize = 256
    
    allocaBytes bufSize $ \outBuf ->
        BS.unsafeUseAsCStringLen nameBs $ \(namePtr, nameLen) -> do
            actualLen <- c_get_out_path
                (castPtr namePtr) (fromIntegral nameLen)
                outBuf (fromIntegral bufSize)
            
            if actualLen == 0
                then error $ "getOutPath failed for: " <> T.unpack name
                else do
                    bs <- BS.packCStringLen (castPtr outBuf, fromIntegral actualLen)
                    pure (T.decodeUtf8 bs)
