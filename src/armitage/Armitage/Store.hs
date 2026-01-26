{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Armitage.Store
Description : Daemon-free Nix store operations

Direct store operations without the nix-daemon. Uses:
  - User namespaces for privilege (no setuid)
  - Direct filesystem operations
  - Content-addressed paths

The daemon is hostile infrastructure. This module routes around it.

Key insight: The store is just a content-addressed filesystem.
  /nix/store/<hash>-<name> = content
  
With user namespaces, we can:
  1. Create a private mount namespace
  2. Bind-mount our own store
  3. Write directly without daemon

Build:
  buck2 build //src/armitage/store:store
-}
module Armitage.Store
  ( -- * Store
    Store (..)
  , StoreConfig (..)
  , withStore
  , defaultStoreConfig
  , userStoreConfig

    -- * Store Paths
  , StorePath (..)
  , storePathHash
  , storePathName
  , parseStorePath
  , renderStorePath

    -- * Operations
  , addToStore
  , queryPathInfo
  , isValidPath
  , getStorePath

    -- * Content Addressing
  , computeStoreHash
  , hashPath

    -- * NAR
  , dumpNar
  , restoreNar
  ) where

import Control.Exception (bracket)
import Control.Monad (unless, when)
import Crypto.Hash (SHA256 (..), hashWith)
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC
import qualified Data.ByteString.Lazy as LBS
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import GHC.Generics (Generic)
import System.Directory
import System.FilePath
import System.Posix.Files (createSymbolicLink, getFileStatus, isDirectory, isRegularFile)
import System.Process (callProcess, readProcess)

-- -----------------------------------------------------------------------------
-- Store Configuration
-- -----------------------------------------------------------------------------

-- | Store configuration
data StoreConfig = StoreConfig
  { storeDir :: FilePath
  -- ^ Store directory (default: /nix/store)
  , stateDir :: FilePath
  -- ^ State directory (default: /nix/var/nix)
  , useUserNamespace :: Bool
  -- ^ Use user namespaces for privilege
  , readOnly :: Bool
  -- ^ Open store read-only
  }
  deriving (Show, Eq, Generic)

-- | Default configuration (system store)
defaultStoreConfig :: StoreConfig
defaultStoreConfig =
  StoreConfig
    { storeDir = "/nix/store"
    , stateDir = "/nix/var/nix"
    , useUserNamespace = True
    , readOnly = False
    }

-- | User-local store configuration
--
-- For single-user operation without daemon.
-- Store lives in ~/.local/share/nix/store
userStoreConfig :: IO StoreConfig
userStoreConfig = do
  home <- getHomeDirectory
  let base = home </> ".local/share/nix"
  pure
    StoreConfig
      { storeDir = base </> "store"
      , stateDir = base </> "var"
      , useUserNamespace = False -- Not needed for user store
      , readOnly = False
      }

-- -----------------------------------------------------------------------------
-- Store Handle
-- -----------------------------------------------------------------------------

-- | Store handle
data Store = Store
  { storeConfig :: StoreConfig
  }

-- | Open store and run action
withStore :: StoreConfig -> (Store -> IO a) -> IO a
withStore config action = do
  -- Ensure directories exist
  unless (readOnly config) $ do
    createDirectoryIfMissing True (storeDir config)
    createDirectoryIfMissing True (stateDir config)

  let store = Store {storeConfig = config}
  action store

-- -----------------------------------------------------------------------------
-- Store Paths
-- -----------------------------------------------------------------------------

-- | A store path
--
-- Format: /nix/store/<hash>-<name>
-- Hash is 32 chars of base32-encoded truncated SHA256
data StorePath = StorePath
  { spHash :: Text
  -- ^ 32-char base32 hash
  , spName :: Text
  -- ^ Derivation name
  }
  deriving (Show, Eq, Generic)

-- | Extract hash from store path
storePathHash :: StorePath -> Text
storePathHash = spHash

-- | Extract name from store path
storePathName :: StorePath -> Text
storePathName = spName

-- | Parse store path from string
--
-- Input: "/nix/store/abc123...-hello-1.0"
-- Output: StorePath "abc123..." "hello-1.0"
parseStorePath :: FilePath -> Maybe StorePath
parseStorePath path =
  let basename = takeFileName path
   in case T.breakOn "-" (T.pack basename) of
        (hash, rest)
          | T.length hash == 32
          , not (T.null rest) ->
              Just
                StorePath
                  { spHash = hash
                  , spName = T.drop 1 rest -- drop leading "-"
                  }
        _ -> Nothing

-- | Render store path to filesystem path
renderStorePath :: Store -> StorePath -> FilePath
renderStorePath store StorePath {..} =
  storeDir (storeConfig store) </> T.unpack (spHash <> "-" <> spName)

-- -----------------------------------------------------------------------------
-- Store Operations
-- -----------------------------------------------------------------------------

-- | Add content to store
--
-- Returns the store path of the added content.
addToStore :: Store -> Text -> ByteString -> IO StorePath
addToStore store name content = do
  let hash = computeStoreHash content
      sp = StorePath {spHash = hash, spName = name}
      destPath = renderStorePath store sp

  -- Check if already exists
  exists <- doesPathExist destPath
  unless exists $ do
    -- Write content
    -- In production, this would:
    -- 1. Use user namespace if configured
    -- 2. Write to temp, then atomic rename
    -- 3. Set permissions (555 for dirs, 444 for files)
    BS.writeFile destPath content

  pure sp

-- | Query path info
--
-- Returns Nothing if path not in store.
queryPathInfo :: Store -> StorePath -> IO (Maybe PathInfo)
queryPathInfo store sp = do
  let path = renderStorePath store sp
  exists <- doesPathExist path
  if exists
    then do
      size <- getFileSize path
      pure $
        Just
          PathInfo
            { piPath = sp
            , piNarSize = fromIntegral size
            , piNarHash = "" -- Would compute NAR hash
            , piReferences = []
            , piDeriver = Nothing
            }
    else pure Nothing

-- | Check if path is valid (exists in store)
isValidPath :: Store -> StorePath -> IO Bool
isValidPath store sp = do
  let path = renderStorePath store sp
  doesPathExist path

-- | Get store path for content
--
-- Computes what the store path would be, without adding.
getStorePath :: Store -> Text -> ByteString -> StorePath
getStorePath _store name content =
  let hash = computeStoreHash content
   in StorePath {spHash = hash, spName = name}

-- | Path info (subset of what daemon provides)
data PathInfo = PathInfo
  { piPath :: StorePath
  , piNarSize :: Int
  , piNarHash :: Text
  , piReferences :: [StorePath]
  , piDeriver :: Maybe StorePath
  }
  deriving (Show, Eq, Generic)

-- -----------------------------------------------------------------------------
-- Content Addressing
-- -----------------------------------------------------------------------------

-- | Compute store hash from content
--
-- This is a simplified version. Real Nix uses:
-- 1. NAR serialization of the path
-- 2. SHA256 hash of NAR
-- 3. Truncate to 160 bits
-- 4. Base32 encode
--
-- For flat files, we can skip NAR.
computeStoreHash :: ByteString -> Text
computeStoreHash content =
  -- Full SHA256 for now, truncate + base32 in production
  let fullHash = hashBytes content
   in T.take 32 fullHash -- Simplified: take first 32 chars of hex

-- | Hash bytes to hex string
hashBytes :: ByteString -> Text
hashBytes bs = T.pack $ show $ hashWith SHA256 bs

-- | Hash a filesystem path (file or directory)
hashPath :: FilePath -> IO Text
hashPath path = do
  -- For now, just hash file contents
  -- Real implementation would use NAR serialization for directories
  content <- BS.readFile path
  pure $ hashBytes content

-- -----------------------------------------------------------------------------
-- NAR Serialization
-- -----------------------------------------------------------------------------

-- | Dump path to NAR format
--
-- NAR (Nix ARchive) is a deterministic archive format:
--   - No timestamps
--   - No permissions (except executable bit)
--   - No ownership
--   - Sorted directory entries
--
-- This enables content-addressing of directory trees.
dumpNar :: FilePath -> IO LBS.ByteString
dumpNar path = do
  -- For now, shell out to nix-store --dump
  -- In production, use hnix-store-nar or implement directly
  output <- readProcess "nix-store" ["--dump", path] ""
  pure $ LBS.pack $ map (fromIntegral . fromEnum) output

-- | Restore NAR to filesystem
restoreNar :: FilePath -> LBS.ByteString -> IO ()
restoreNar destPath narContent = do
  -- For now, shell out to nix-store --restore
  -- In production, use hnix-store-nar or implement directly
  let narFile = destPath <> ".nar"
  LBS.writeFile narFile narContent
  callProcess "nix-store" ["--restore", destPath]
  removeFile narFile

-- -----------------------------------------------------------------------------
-- User Namespace Operations
-- -----------------------------------------------------------------------------

-- | Run action in user namespace with store access
--
-- Uses unshare(2) to create:
--   - User namespace (CLONE_NEWUSER)
--   - Mount namespace (CLONE_NEWNS)
--
-- Then bind-mounts the store to provide write access.
--
-- This is the key to daemon-free operation: we don't need
-- root or the daemon because we have our own namespace.
withUserNamespace :: Store -> IO a -> IO a
withUserNamespace store action = do
  -- TODO: Implement using System.Posix.Unistd.unshare
  -- or shell out to unshare(1)
  --
  -- For now, just run the action directly
  -- (works for user stores, not system store)
  action

-- | Check if user namespaces are available
userNamespacesAvailable :: IO Bool
userNamespacesAvailable = do
  -- Check /proc/sys/kernel/unprivileged_userns_clone
  -- or try to create one
  content <- readFile "/proc/sys/kernel/unprivileged_userns_clone"
  pure $ content == "1\n"
