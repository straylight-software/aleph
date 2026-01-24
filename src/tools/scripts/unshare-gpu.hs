{-# LANGUAGE OverloadedStrings #-}

{- |
oci-gpu: Run OCI container images with NVIDIA GPU access

Uses Aleph.Script - batteries-included shell scripting for Haskell.
Much cleaner than raw Turtle or Shelly!
-}
module Main where

import Aleph.Script
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Lazy as LBS
import qualified Data.Text as T
import qualified Data.Vector as V
import qualified System.Environment as Env
import System.Posix.Process (executeFile)
import Prelude hiding (FilePath)

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

data ContainerEnv = ContainerEnv
    { cenvPath :: Maybe Text
    , cenvLdLibPath :: Maybe Text
    }

emptyEnv :: ContainerEnv
emptyEnv = ContainerEnv Nothing Nothing

-- ============================================================================
-- Main
-- ============================================================================

main :: IO ()
main = script $ do
    args <- liftIO Env.getArgs
    case args of
        [] -> die "Usage: oci-gpu IMAGE [COMMAND...]"
        (img : cmdArgs) -> do
            let cmd' = if Prelude.null cmdArgs then ["nvidia-smi"] else Prelude.map pack cmdArgs
            cfg <- buildConfig (pack img) cmd'
            runWithGpu cfg

buildConfig :: Text -> [Text] -> Sh Config
buildConfig img cmd' = do
    homeDir <- getEnvDefault "HOME" "/tmp"
    xdgCache <- getEnv "XDG_CACHE_HOME"
    let cacheBase = fromText $ fromMaybe (homeDir <> "/.cache") xdgCache

    pure
        Config
            { cfgImage = img
            , cfgCommand = cmd'
            , cfgPlatform = "linux/amd64"
            , cfgCacheDir = cacheBase </> "straylight-oci"
            , cfgCertFile = "/etc/ssl/certs/ca-bundle.crt"
            }

-- ============================================================================
-- Core Logic
-- ============================================================================

runWithGpu :: Config -> Sh ()
runWithGpu cfg = do
    mkdirP (cfgCacheDir cfg)

    cacheKey <- getCacheKey (cfgImage cfg)
    let cachedRootfs = cfgCacheDir cfg </> unpack cacheKey

    withTmpDir $ \workDir -> do
        let rootfsLink = workDir </> "rootfs"

        cached <- test_d cachedRootfs
        if cached
            then do
                echoErr $ ":: Using cached " <> cfgImage cfg
                symlink cachedRootfs rootfsLink
            else do
                echoErr $ ":: Pulling " <> cfgImage cfg
                pullImage cfg workDir
                mv (workDir </> "rootfs") cachedRootfs
                symlink cachedRootfs rootfsLink
                echoErr $ ":: Cached to " <> pack cachedRootfs

        mkdirP (rootfsLink </> "usr/local/nvidia/bin")
        mkdirP (rootfsLink </> "usr/local/nvidia/lib64")

        nvBinds <- withGpuBinds
        containerEnv <- getContainerEnv cfg

        let combinedPath = buildPath containerEnv
            combinedLdPath = buildLdPath containerEnv

        echoErr ":: Entering namespace with GPU"
        let bwrapArgs = buildBwrapArgs workDir nvBinds combinedPath combinedLdPath (cfgCommand cfg)

        liftIO $ executeFile "bwrap" True (Prelude.map unpack bwrapArgs) Nothing

-- ============================================================================
-- Image Operations
-- ============================================================================

getCacheKey :: Text -> Sh Text
getCacheKey img = do
    result <- bash $ "echo -n '" <> img <> "' | sha256sum | cut -c1-16"
    pure $ strip result

pullImage :: Config -> FilePath -> Sh ()
pullImage cfg workDir = do
    let rootfs = workDir </> "rootfs"
    mkdirP rootfs
    setEnv "SSL_CERT_FILE" (pack $ cfgCertFile cfg)
    bash_ $
        "crane export --platform "
            <> cfgPlatform cfg
            <> " '"
            <> cfgImage cfg
            <> "' - | tar -xf - -C "
            <> pack rootfs

getContainerEnv :: Config -> Sh ContainerEnv
getContainerEnv cfg = do
    result <- errExit False $ run "crane" ["config", cfgImage cfg]
    code <- exitCode
    if code == 0
        then pure $ parseEnvFromJson result
        else pure emptyEnv
  where
    parseEnvFromJson :: Text -> ContainerEnv
    parseEnvFromJson json =
        case decode (LBS.fromStrict $ encodeUtf8 json) :: Maybe Value of
            Just val -> extractEnv val
            Nothing -> emptyEnv

    extractEnv :: Value -> ContainerEnv
    extractEnv val = fromMaybe emptyEnv $ do
        Object root <- Just val
        Object cfgObj <- KM.lookup "config" root
        Array envArr <- KM.lookup "Env" cfgObj
        let envPairs = Prelude.map extractEnvVar (V.toList envArr)
        Just
            ContainerEnv
                { cenvPath = Prelude.lookup "PATH" envPairs
                , cenvLdLibPath = Prelude.lookup "LD_LIBRARY_PATH" envPairs
                }

    extractEnvVar (String str) = let (k, v) = breakOn "=" str in (k, T.drop 1 v)
    extractEnvVar _ = ("", "")

-- ============================================================================
-- Environment Building
-- ============================================================================

buildPath :: ContainerEnv -> Text
buildPath env =
    "/usr/local/nvidia/bin:" <> fromMaybe defaultPath (cenvPath env)
  where
    defaultPath = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

buildLdPath :: ContainerEnv -> Text
buildLdPath env =
    "/usr/local/nvidia/lib64:/run/opengl-driver/lib"
        <> maybe "" (":" <>) (cenvLdLibPath env)

-- ============================================================================
-- Bwrap Execution
-- ============================================================================

buildBwrapArgs :: FilePath -> [Text] -> Text -> Text -> [Text] -> [Text]
buildBwrapArgs workDir nvBinds path ldPath cmd' =
    [ "--bind"
    , pack (workDir </> "rootfs")
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
           , path
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
