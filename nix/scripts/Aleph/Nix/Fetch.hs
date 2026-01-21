{-# LANGUAGE CApiFFI #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE OverloadedStrings #-}

{- | Fetch primitives - call INTO Nix to fetch sources.

This inverts the control flow: Haskell calls Nix for fetching,
rather than Nix calling Haskell. The WASI host exposes these
primitives, and Haskell orchestrates the build.

= FFI to straylight-nix host

The host provides:
  - fetch_github: Fetch from GitHub, return store path
  - fetch_url: Fetch URL, return store path
  - fetch_git: Fetch git repo, return store path

All functions are synchronous from WASM's perspective - the host
handles the actual async I/O and caching.

= Usage

@
import Aleph.Nix.Fetch

main = do
  -- Fetch zlib-ng source
  srcPath <- fetchGitHub "zlib-ng" "zlib-ng" "2.2.4" 
               "sha256-Khmrhp5qy4vvoQe4WgoogpjWrgcUB/q8zZeqIydthYg="
  
  -- srcPath is now a store path like /nix/store/abc123-source
  putStrLn $ "Source at: " <> srcPath
@
-}
module Aleph.Nix.Fetch (
    -- * GitHub
    fetchGitHub,
    
    -- * URLs
    fetchUrl,
    fetchTarball,
    
    -- * Git
    fetchGit,
    
    -- * Low-level FFI
    c_fetch_github,
    c_fetch_url,
    c_fetch_git,
) where

import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.ByteString as BS
import qualified Data.ByteString.Unsafe as BS
import Data.Word (Word8)
import Foreign.C.Types (CSize (..))
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Marshal.Alloc (allocaBytes)

--------------------------------------------------------------------------------
-- Low-level FFI
--------------------------------------------------------------------------------

-- | Fetch from GitHub. Returns store path.
-- Parameters: owner, owner_len, repo, repo_len, rev, rev_len, hash, hash_len
-- Returns: pointer to store path string (null-terminated)
foreign import ccall unsafe "nix_fetch_github"
    c_fetch_github :: Ptr Word8 -> CSize   -- owner
                   -> Ptr Word8 -> CSize   -- repo
                   -> Ptr Word8 -> CSize   -- rev
                   -> Ptr Word8 -> CSize   -- hash (SRI format)
                   -> Ptr Word8 -> CSize   -- output buffer
                   -> IO CSize             -- actual length

-- | Fetch URL. Returns store path.
foreign import ccall unsafe "nix_fetch_url"
    c_fetch_url :: Ptr Word8 -> CSize      -- url
                -> Ptr Word8 -> CSize      -- hash (SRI format)
                -> Ptr Word8 -> CSize      -- output buffer
                -> IO CSize                -- actual length

-- | Fetch git repo. Returns store path.
foreign import ccall unsafe "nix_fetch_git"
    c_fetch_git :: Ptr Word8 -> CSize      -- url
                -> Ptr Word8 -> CSize      -- rev
                -> Ptr Word8 -> CSize      -- hash (SRI format)
                -> Ptr Word8 -> CSize      -- output buffer
                -> IO CSize                -- actual length

--------------------------------------------------------------------------------
-- High-level API
--------------------------------------------------------------------------------

-- | Fetch a GitHub repository.
--
-- @
-- srcPath <- fetchGitHub "zlib-ng" "zlib-ng" "2.2.4" "sha256-..."
-- @
fetchGitHub :: Text    -- ^ Owner
            -> Text    -- ^ Repository name
            -> Text    -- ^ Revision (tag, branch, or commit)
            -> Text    -- ^ SRI hash (e.g., "sha256-...")
            -> IO Text -- ^ Store path
fetchGitHub owner repo rev hash = do
    let ownerBs = T.encodeUtf8 owner
        repoBs = T.encodeUtf8 repo
        revBs = T.encodeUtf8 rev
        hashBs = T.encodeUtf8 hash
        bufSize = 256  -- Store paths are typically ~50 chars
    
    allocaBytes bufSize $ \outBuf ->
        BS.unsafeUseAsCStringLen ownerBs $ \(ownerPtr, ownerLen) ->
        BS.unsafeUseAsCStringLen repoBs $ \(repoPtr, repoLen) ->
        BS.unsafeUseAsCStringLen revBs $ \(revPtr, revLen) ->
        BS.unsafeUseAsCStringLen hashBs $ \(hashPtr, hashLen) -> do
            actualLen <- c_fetch_github
                (castPtr ownerPtr) (fromIntegral ownerLen)
                (castPtr repoPtr) (fromIntegral repoLen)
                (castPtr revPtr) (fromIntegral revLen)
                (castPtr hashPtr) (fromIntegral hashLen)
                outBuf (fromIntegral bufSize)
            
            if actualLen == 0
                then error $ "fetchGitHub failed for " <> T.unpack owner <> "/" <> T.unpack repo
                else do
                    bs <- BS.packCStringLen (castPtr outBuf, fromIntegral actualLen)
                    pure (T.decodeUtf8 bs)

-- | Fetch a URL.
--
-- @
-- srcPath <- fetchUrl "https://example.com/foo.tar.gz" "sha256-..."
-- @
fetchUrl :: Text    -- ^ URL
         -> Text    -- ^ SRI hash
         -> IO Text -- ^ Store path
fetchUrl url hash = do
    let urlBs = T.encodeUtf8 url
        hashBs = T.encodeUtf8 hash
        bufSize = 256
    
    allocaBytes bufSize $ \outBuf ->
        BS.unsafeUseAsCStringLen urlBs $ \(urlPtr, urlLen) ->
        BS.unsafeUseAsCStringLen hashBs $ \(hashPtr, hashLen) -> do
            actualLen <- c_fetch_url
                (castPtr urlPtr) (fromIntegral urlLen)
                (castPtr hashPtr) (fromIntegral hashLen)
                outBuf (fromIntegral bufSize)
            
            if actualLen == 0
                then error $ "fetchUrl failed for " <> T.unpack url
                else do
                    bs <- BS.packCStringLen (castPtr outBuf, fromIntegral actualLen)
                    pure (T.decodeUtf8 bs)

-- | Fetch a tarball URL (same as fetchUrl but semantically different).
fetchTarball :: Text -> Text -> IO Text
fetchTarball = fetchUrl

-- | Fetch a git repository.
--
-- @
-- srcPath <- fetchGit "https://github.com/foo/bar.git" "v1.0.0" "sha256-..."
-- @
fetchGit :: Text    -- ^ Git URL
         -> Text    -- ^ Revision
         -> Text    -- ^ SRI hash
         -> IO Text -- ^ Store path
fetchGit url rev hash = do
    let urlBs = T.encodeUtf8 url
        revBs = T.encodeUtf8 rev
        hashBs = T.encodeUtf8 hash
        bufSize = 256
    
    allocaBytes bufSize $ \outBuf ->
        BS.unsafeUseAsCStringLen urlBs $ \(urlPtr, urlLen) ->
        BS.unsafeUseAsCStringLen revBs $ \(revPtr, revLen) ->
        BS.unsafeUseAsCStringLen hashBs $ \(hashPtr, hashLen) -> do
            actualLen <- c_fetch_git
                (castPtr urlPtr) (fromIntegral urlLen)
                (castPtr revPtr) (fromIntegral revLen)
                (castPtr hashPtr) (fromIntegral hashLen)
                outBuf (fromIntegral bufSize)
            
            if actualLen == 0
                then error $ "fetchGit failed for " <> T.unpack url
                else do
                    bs <- BS.packCStringLen (castPtr outBuf, fromIntegral actualLen)
                    pure (T.decodeUtf8 bs)
