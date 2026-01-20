{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Oci
Description : OCI container operations - pulling, caching, running

Common infrastructure for OCI container scripts. Handles:

* Image pulling and caching
* Container environment extraction
* Bwrap sandbox construction

== Example

@
import Aleph.Script
import qualified Aleph.Script.Oci as Oci
import qualified Aleph.Script.Tools.Bwrap as Bwrap

main = script $ do
  -- Pull and cache an image
  rootfs <- Oci.pullOrCache Oci.defaultConfig "alpine:latest"

  -- Build sandbox and run
  let sandbox = Oci.baseSandbox rootfs
  Bwrap.exec sandbox ["/bin/sh"]
@
-}
module Aleph.Script.Oci (
    -- * Configuration
    Config (..),
    defaultConfig,

    -- * Image operations
    pullOrCache,
    computeCacheKey,
    getContainerEnv,
    ContainerEnv (..),
    emptyEnv,

    -- * Sandbox construction
    baseSandbox,
    withGpuSupport,

    -- * Path building
    buildPath,
    buildLdPath,
) where

import Aleph.Script hiding (FilePath)
import qualified Aleph.Script.Tools.Bwrap as Bwrap
import qualified Aleph.Script.Tools.Crane as Crane
import Crypto.Hash (SHA256 (..), hashWith)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Lazy as BL
import Data.Function ((&))
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Vector as V

-- ============================================================================
-- Types
-- ============================================================================

-- | OCI runtime configuration
data Config = Config
    { cfgPlatform :: Crane.Platform
    -- ^ Target platform
    , cfgCacheDir :: FilePath
    -- ^ Cache directory for extracted images
    , cfgCertFile :: FilePath
    -- ^ SSL certificate file
    }
    deriving (Show)

-- | Default configuration
defaultConfig :: Config
defaultConfig =
    Config
        { cfgPlatform = Crane.defaultPlatform
        , cfgCacheDir = "" -- Will be set based on XDG_CACHE_HOME
        , cfgCertFile = "/etc/ssl/certs/ca-bundle.crt"
        }

-- | Environment variables extracted from container config
data ContainerEnv = ContainerEnv
    { cePath :: Maybe Text
    -- ^ PATH from container
    , ceLdLibPath :: Maybe Text
    -- ^ LD_LIBRARY_PATH from container
    }
    deriving (Show)

-- | Empty container environment
emptyEnv :: ContainerEnv
emptyEnv = ContainerEnv Nothing Nothing

-- ============================================================================
-- Image Operations
-- ============================================================================

-- | Compute cache key from image name (SHA256, first 16 chars)
computeCacheKey :: Text -> Text
computeCacheKey image =
    let hash = hashWith SHA256 (TE.encodeUtf8 image)
     in T.take 16 $ pack $ show hash

{- | Pull an image or use cached version, returns path to rootfs

The rootfs is a symlink to the cached extracted image.
-}
pullOrCache :: Config -> Text -> Sh FilePath
pullOrCache cfg image = do
    -- Resolve cache directory
    cacheDir <-
        if cfgCacheDir cfg == ""
            then do
                homeDir <- getEnvDefault "HOME" "/root"
                xdgCache <- getEnv "XDG_CACHE_HOME"
                pure $ case xdgCache of
                    Just c -> unpack c </> "straylight-oci"
                    Nothing -> unpack homeDir </> ".cache" </> "straylight-oci"
            else pure (cfgCacheDir cfg)

    mkdirP cacheDir

    let cacheKey = computeCacheKey image
        cachedRootfs = cacheDir </> unpack cacheKey

    -- Check cache
    cached <- test_d cachedRootfs
    if cached
        then do
            echoErr $ ":: Using cached " <> image
            pure cachedRootfs
        else do
            echoErr $ ":: Pulling " <> image

            -- Set SSL cert
            setEnv "SSL_CERT_FILE" (pack $ cfgCertFile cfg)

            -- Pull to temp dir then move to cache
            -- Use cp + rm instead of mv to handle cross-device moves
            -- (e.g., /tmp on tmpfs, ~/.cache on disk)
            withTmpDir $ \tmpDir -> do
                let tmpRootfs = tmpDir </> "rootfs"
                let opts = Crane.defaults{Crane.platform_ = Just (cfgPlatform cfg)}
                Crane.exportToDir opts image tmpRootfs
                -- cp -a preserves permissions/ownership, works across filesystems
                run_ "cp" ["-a", pack tmpRootfs, pack cachedRootfs]
                echoErr $ ":: Cached to " <> pack cachedRootfs
                pure cachedRootfs

-- | Get container environment from image config
getContainerEnv :: Text -> Sh ContainerEnv
getContainerEnv image = do
    configJson <- errExit False $ Crane.config image
    code <- exitCode

    if code /= 0
        then pure emptyEnv
        else case Aeson.eitherDecode (BL.fromStrict $ TE.encodeUtf8 configJson) of
            Left _ -> pure emptyEnv
            Right val -> pure $ parseEnvFromConfig val

-- | Parse environment variables from OCI image config JSON
parseEnvFromConfig :: Aeson.Value -> ContainerEnv
parseEnvFromConfig val = case val of
    Aeson.Object obj ->
        case KM.lookup "config" obj of
            Just (Aeson.Object cfgObj) ->
                case KM.lookup "Env" cfgObj of
                    Just (Aeson.Array envArr) ->
                        let envPairs = map extractEnvVar (V.toList envArr)
                         in ContainerEnv
                                { cePath = lookup "PATH" envPairs
                                , ceLdLibPath = lookup "LD_LIBRARY_PATH" envPairs
                                }
                    _ -> emptyEnv
            _ -> emptyEnv
    _ -> emptyEnv
  where
    extractEnvVar :: Aeson.Value -> (Text, Text)
    extractEnvVar (Aeson.String str) =
        let (k, v) = breakOn "=" str
         in (k, T.drop 1 v)
    extractEnvVar _ = ("", "")

-- ============================================================================
-- Sandbox Construction
-- ============================================================================

-- | Build a base sandbox for running containers (no GPU)
baseSandbox :: FilePath -> Bwrap.Sandbox
baseSandbox rootfs =
    Bwrap.defaults
        & Bwrap.bind rootfs "/"
        & Bwrap.dev "/dev"
        & Bwrap.proc "/proc"
        & Bwrap.tmpfs "/tmp"
        & Bwrap.tmpfs "/run"
        & Bwrap.roBind "/etc/resolv.conf" "/etc/resolv.conf"
        & Bwrap.roBind "/etc/ssl" "/etc/ssl"
        & Bwrap.setenv "PATH" "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        & Bwrap.setenv "HOME" "/root"
        & Bwrap.chdir "/root"
        & Bwrap.dieWithParent
        & Bwrap.unsharePid

-- | Add GPU support to a sandbox
withGpuSupport :: ContainerEnv -> [Text] -> Bwrap.Sandbox -> Bwrap.Sandbox
withGpuSupport env gpuBinds sandbox =
    sandbox
        & Bwrap.roBind "/sys" "/sys"
        & applyGpuBinds gpuBinds
        & Bwrap.setenv "PATH" (buildPath env)
        & Bwrap.setenv "LD_LIBRARY_PATH" (buildLdPath env)
        & Bwrap.setenv "OPAL_PREFIX" "/opt/hpcx/ompi"
        & Bwrap.setenv "OMPI_MCA_btl" "^openib"

-- | Apply GPU bind mounts to a sandbox
applyGpuBinds :: [Text] -> Bwrap.Sandbox -> Bwrap.Sandbox
applyGpuBinds binds sandbox = go binds sandbox
  where
    go [] s = s
    go ("--dev-bind" : src : dst : rest) s = go rest (Bwrap.devBind (unpack src) (unpack dst) s)
    go ("--ro-bind" : src : dst : rest) s = go rest (Bwrap.roBind (unpack src) (unpack dst) s)
    go (_ : rest) s = go rest s

-- ============================================================================
-- Path Building
-- ============================================================================

-- | Build PATH for container with nvidia
buildPath :: ContainerEnv -> Text
buildPath ContainerEnv{..} =
    let base = "/usr/local/nvidia/bin"
        defaultPath = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
     in case cePath of
            Just p -> base <> ":" <> p
            Nothing -> base <> ":" <> defaultPath

-- | Build LD_LIBRARY_PATH for container with nvidia
buildLdPath :: ContainerEnv -> Text
buildLdPath ContainerEnv{..} =
    let base = "/usr/local/nvidia/lib64:/run/opengl-driver/lib"
     in case ceLdLibPath of
            Just p -> base <> ":" <> p
            Nothing -> base
