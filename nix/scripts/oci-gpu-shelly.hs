{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE OverloadedStrings #-}
{-# OPTIONS_GHC -fno-warn-type-defaults #-}

{- |
Module      : OciGpu
Description : Run OCI container images with NVIDIA GPU access

A typed shell script that pulls OCI images, caches them, and runs them
in a bubblewrap namespace with GPU device passthrough.

This is the Shelly version. Compared to Turtle:
  - Thread-safe (maintains its own environment state)
  - Command tracing built-in (great for debugging)
  - More verbose API but more control
  - Uses monad transformer (ShIO)
-}
module Main where

import Data.Aeson (Value (..), decode, (.:?))
import Data.Aeson.Types (parseMaybe)
import qualified Data.Aeson.Types as Aeson
import qualified Data.ByteString.Lazy as BL
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Vector as V
import Shelly
import qualified System.Environment
import System.FilePath (takeFileName)
import System.Posix.Process (executeFile)

default (Text)

-- ============================================================================
-- Types
-- ============================================================================

data Config = Config
    { cfgImage :: Text
    , cfgCommand :: [Text]
    , cfgPlatform :: Text
    , cfgCacheDir :: FilePath
    , cfgCertFile :: FilePath
    }
    deriving (Show)

data ContainerEnv = ContainerEnv
    { envPath :: Maybe Text
    , envLdLibPath :: Maybe Text
    }
    deriving (Show)

emptyEnv :: ContainerEnv
emptyEnv = ContainerEnv Nothing Nothing

-- ============================================================================
-- Main
-- ============================================================================

main :: IO ()
main = shelly $ verbosely $ do
    args <- liftIO System.Environment.getArgs
    case args of
        [] -> errorExit "Usage: oci-gpu IMAGE [COMMAND...]"
        (image : cmdArgs) -> do
            let cmd' = if null cmdArgs then ["nvidia-smi"] else map T.pack cmdArgs
            cfg <- buildConfig (T.pack image) cmd'
            runWithGpu cfg

buildConfig :: Text -> [Text] -> Sh Config
buildConfig image cmd' = do
    homeDir <- fromMaybe (error "HOME not set") <$> get_env "HOME"
    xdgCache <- get_env "XDG_CACHE_HOME"
    let cacheBase = case xdgCache of
            Just c -> fromText c
            Nothing -> fromText homeDir </> ".cache"

    -- These would be injected by Nix in the real build
    let certFile = "/etc/ssl/certs/ca-bundle.crt"
        platform = "linux/amd64"

    pure
        Config
            { cfgImage = image
            , cfgCommand = cmd'
            , cfgPlatform = platform
            , cfgCacheDir = cacheBase </> "straylight-oci"
            , cfgCertFile = fromText certFile
            }

-- ============================================================================
-- Core Logic
-- ============================================================================

runWithGpu :: Config -> Sh ()
runWithGpu cfg = do
    -- Ensure cache directory exists
    mkdir_p (cfgCacheDir cfg)

    -- Compute cache key from image name
    cacheKey <- getCacheKey (cfgImage cfg)
    let cachedRootfs = cfgCacheDir cfg </> fromText cacheKey

    -- Create temp working directory
    -- Shelly has withTmpDir for automatic cleanup
    withTmpDir $ \workDir -> do
        let rootfsLink = workDir </> "rootfs"

        -- Pull or use cached image
        cached <- test_d cachedRootfs
        if cached
            then do
                echo $ ":: Using cached " <> cfgImage cfg
                -- Shelly doesn't have symlink, use command
                run_ "ln" ["-s", toTextIgnore cachedRootfs, toTextIgnore rootfsLink]
            else do
                echo $ ":: Pulling " <> cfgImage cfg
                pullImage cfg workDir
                mv (workDir </> "rootfs") cachedRootfs
                run_ "ln" ["-s", toTextIgnore cachedRootfs, toTextIgnore rootfsLink]
                echo $ ":: Cached to " <> toTextIgnore cachedRootfs

        -- Create nvidia mount points
        mkdir_p (rootfsLink </> "usr/local/nvidia/bin")
        mkdir_p (rootfsLink </> "usr/local/nvidia/lib64")

        -- Discover GPU devices and drivers
        nvBinds <- discoverNvidiaBinds

        -- Extract container environment from image config
        containerEnv <- getContainerEnv cfg

        -- Build final environment
        let combinedPath = buildPath containerEnv
            combinedLdPath = buildLdPath containerEnv

        -- Execute bwrap
        echo ":: Entering namespace with GPU"
        let bwrapArgs = buildBwrapArgs workDir nvBinds combinedPath combinedLdPath (cfgCommand cfg)

        -- Use exec to replace process
        liftIO $ executeFile "bwrap" True (map T.unpack bwrapArgs) Nothing

-- ============================================================================
-- Image Operations
-- ============================================================================

getCacheKey :: Text -> Sh Text
getCacheKey image = do
    -- SHA256 hash of image name, first 16 chars
    -- Shelly's escaping False equivalent is using bash directly
    result <- run "sh" ["-c", "echo -n '" <> image <> "' | sha256sum | cut -c1-16"]
    pure $ T.strip result

pullImage :: Config -> FilePath -> Sh ()
pullImage cfg workDir = do
    let rootfs = workDir </> "rootfs"
    mkdir_p rootfs

    -- Set SSL cert for crane
    setenv "SSL_CERT_FILE" (toTextIgnore $ cfgCertFile cfg)

    -- Pull and extract in a pipeline
    run_
        "sh"
        [ "-c"
        , "crane export --platform " <> cfgPlatform cfg <> " '" <> cfgImage cfg <> "' - | tar -xf - -C " <> toTextIgnore rootfs
        ]

getContainerEnv :: Config -> Sh ContainerEnv
getContainerEnv cfg = do
    -- Run crane config and parse JSON
    -- errExit False to not fail on non-zero exit
    result <- errExit False $ run "crane" ["config", cfgImage cfg]
    code <- lastExitCode
    if code == 0
        then case decode (BL.fromStrict $ TE.encodeUtf8 result) of
            Just val -> pure $ parseEnvFromConfig val
            Nothing -> pure emptyEnv
        else pure emptyEnv
  where
    parseEnvFromConfig :: Value -> ContainerEnv
    parseEnvFromConfig val = fromMaybe emptyEnv $ parseMaybe parseConfig val

    parseConfig :: Value -> Aeson.Parser ContainerEnv
    parseConfig (Object obj) = do
        mConfig <- obj .:? "config"
        case mConfig of
            Just (Object cfgObj) -> do
                mEnvList <- cfgObj .:? "Env"
                case mEnvList of
                    Just (Array arr) -> do
                        let envPairs = map extractEnvVar (V.toList arr)
                        pure
                            ContainerEnv
                                { envPath = lookup "PATH" envPairs
                                , envLdLibPath = lookup "LD_LIBRARY_PATH" envPairs
                                }
                    _ -> pure emptyEnv
            _ -> pure emptyEnv
    parseConfig _ = pure emptyEnv

    extractEnvVar :: Value -> (Text, Text)
    extractEnvVar (String str) =
        let (k, v) = T.breakOn "=" str
         in (k, T.drop 1 v)
    extractEnvVar _ = ("", "")

-- ============================================================================
-- NVIDIA Discovery
-- ============================================================================

discoverNvidiaBinds :: Sh [Text]
discoverNvidiaBinds = do
    devBinds <- discoverDevices
    driverBinds <- discoverDriver
    glBinds <- discoverOpenGL
    let nixBind = ["--ro-bind", "/nix/store", "/nix/store"]
    pure $ devBinds <> driverBinds <> glBinds <> nixBind

discoverDevices :: Sh [Text]
discoverDevices = do
    -- Find /dev/nvidia* devices
    -- Shelly's findWhen takes a predicate on FilePath
    allDevs <- ls "/dev"
    let nvDevs = filter (T.isPrefixOf "nvidia" . T.pack . takeFileName . T.unpack . toTextIgnore) allDevs

    -- Find /dev/dri/* devices
    driExists <- test_d "/dev/dri"
    driDevs <-
        if driExists
            then ls "/dev/dri"
            else pure []

    let allBindDevs = nvDevs <> driDevs
        binds = concatMap (\dev -> ["--dev-bind", toTextIgnore dev, toTextIgnore dev]) allBindDevs
    pure binds

discoverDriver :: Sh [Text]
discoverDriver = do
    -- Find nvidia driver path via nvidia-smi symlink
    result <- errExit False $ run "readlink" ["-f", "/run/current-system/sw/bin/nvidia-smi"]
    code <- lastExitCode
    if code == 0
        then do
            let driverPath = T.replace "/bin/nvidia-smi" "" (T.strip result)
                driverDir = fromText driverPath

            exists <- test_d driverDir
            if exists
                then do
                    echo $ ":: Found nvidia driver at " <> driverPath
                    binBind <- ifDirExists (driverDir </> "bin") "/usr/local/nvidia/bin"
                    libBind <- ifDirExists (driverDir </> "lib") "/usr/local/nvidia/lib64"
                    pure $ binBind <> libBind
                else pure []
        else pure []
  where
    ifDirExists :: FilePath -> Text -> Sh [Text]
    ifDirExists src dst = do
        exists <- test_d src
        if exists
            then pure ["--ro-bind", toTextIgnore src, dst]
            else pure []

discoverOpenGL :: Sh [Text]
discoverOpenGL = do
    -- Bind opengl drivers if present
    gl1 <- glBind "/run/opengl-driver"
    gl2 <- glBind "/run/opengl-driver-32"
    pure $ gl1 <> gl2
  where
    glBind :: FilePath -> Sh [Text]
    glBind glPath = do
        exists <- test_d glPath
        if exists
            then pure ["--ro-bind", toTextIgnore glPath, toTextIgnore glPath]
            else pure []

-- ============================================================================
-- Environment Building
-- ============================================================================

buildPath :: ContainerEnv -> Text
buildPath cenv =
    let base = "/usr/local/nvidia/bin"
        defaultPath = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
     in case envPath cenv of
            Just p -> base <> ":" <> p
            Nothing -> base <> ":" <> defaultPath

buildLdPath :: ContainerEnv -> Text
buildLdPath cenv =
    let base = "/usr/local/nvidia/lib64:/run/opengl-driver/lib"
     in case envLdLibPath cenv of
            Just p -> base <> ":" <> p
            Nothing -> base

-- ============================================================================
-- Bwrap Execution
-- ============================================================================

buildBwrapArgs :: FilePath -> [Text] -> Text -> Text -> [Text] -> [Text]
buildBwrapArgs workDir nvBinds envPath ldPath cmd' =
    [ "--bind"
    , toTextIgnore (workDir </> "rootfs")
    , "/"
    , "--dev"
    , "/dev"
    , "--proc"
    , "/proc"
    , "--ro-bind"
    , "/sys"
    , "/sys"
    , "--tmpfs"
    , "/tmp"
    , "--tmpfs"
    , "/run"
    ]
        <> nvBinds
        <> [ "--ro-bind"
           , "/etc/resolv.conf"
           , "/etc/resolv.conf"
           , "--ro-bind"
           , "/etc/ssl"
           , "/etc/ssl"
           , "--setenv"
           , "PATH"
           , envPath
           , "--setenv"
           , "HOME"
           , "/root"
           , "--setenv"
           , "LD_LIBRARY_PATH"
           , ldPath
           , "--setenv"
           , "OPAL_PREFIX"
           , "/opt/hpcx/ompi"
           , "--setenv"
           , "OMPI_MCA_btl"
           , "^openib"
           , "--chdir"
           , "/root"
           , "--die-with-parent"
           , "--unshare-pid"
           , "--"
           ]
        <> cmd'
