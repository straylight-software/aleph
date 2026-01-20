{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

-- \|
-- nvidia-extract - Extract NVIDIA SDK components from NGC container
--
-- Usage:
--   nvidia-extract <image-ref> <output-dir>
--
-- Example:
--   nvidia-extract nvcr.io/nvidia/tritonserver:25.11-py3 ./nvidia-sdk
--
-- Extracts:
--   - CUDA toolkit (nvcc, libraries)
--   - cuDNN
--   - NCCL
--   - TensorRT
--   - cuTensor
--
-- Version info is parsed from container environment variables:
--   CUDA_VERSION, CUDNN_VERSION, NCCL_VERSION, TENSORRT_VERSION
--
-- The NGC containers have blessed, tested configurations.
-- No more fighting nvidia's download auth.

import Aleph.Script
import qualified Aleph.Script.Tools.Crane as Crane
import Control.Monad (forM_, when)
import qualified Control.Monad as M
import Data.Aeson (Object, Value (..), decode)
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Lazy as BL
import Data.Maybe (fromMaybe, mapMaybe)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.IO as TIO
import qualified Data.Vector as V
import System.Environment (getArgs)
import Prelude hiding (FilePath)

-- | Version info extracted from container
data NvidiaVersions = NvidiaVersions
    { nvCuda :: Text
    , nvCudnn :: Text
    , nvNccl :: Text
    , nvTensorrt :: Text
    , nvCutensor :: Text
    }
    deriving (Show)

-- | Libraries we care about
targetLibs :: [Text]
targetLibs =
    [ "libcudart"
    , "libcublas"
    , "libcufft"
    , "libcurand"
    , "libcusolver"
    , "libcusparse"
    , "libnvrtc"
    , "libcudnn"
    , "libnccl"
    , "libnvinfer"
    , "libcutensor"
    ]

main :: IO ()
main = script $ verbosely $ do
    args <- liftIO getArgs
    case args of
        [imageRef, outputDir] -> do
            extractNvidiaSdk (pack imageRef) (fromText $ pack outputDir)
        _ -> do
            echoErr "Usage: nvidia-extract <image-ref> <output-dir>"
            echoErr ""
            echoErr "Example:"
            echoErr "  nvidia-extract nvcr.io/nvidia/tritonserver:25.11-py3 ./nvidia-sdk"
            exit 1

extractNvidiaSdk :: Text -> FilePath -> Sh ()
extractNvidiaSdk imageRef outputDir = do
    echoErr $ ":: Extracting NVIDIA SDK from " <> imageRef

    -- Get version info from container config before extraction
    echoErr ":: Reading container config..."
    versions <- getContainerVersions imageRef

    -- Create temp dir for full container
    withTmpDir $ \tmpDir -> do
        let containerRoot = tmpDir </> "rootfs"

        -- Pull and extract container
        echoErr ":: Pulling container..."
        Crane.exportToDir Crane.defaults imageRef containerRoot

        -- Create output structure
        echoErr ":: Creating SDK layout..."
        mkdirP (outputDir </> "bin")
        mkdirP (outputDir </> "lib64")
        mkdirP (outputDir </> "include")
        mkdirP (outputDir </> "nvvm")

        -- Extract CUDA toolkit
        -- Look for versioned cuda dir (e.g., cuda-13.0) since symlinks may not resolve
        cudaDir <- findCudaDir containerRoot (nvCuda versions)
        case cudaDir of
            Nothing -> echoErr ":: Warning: CUDA directory not found"
            Just dir -> do
                echoErr $ ":: Extracting CUDA toolkit " <> nvCuda versions <> " from " <> toTextIgnore dir <> "..."
                -- bin (nvcc, etc)
                copyDir (dir </> "bin") (outputDir </> "bin")
                -- lib64 - try both direct and targets structure
                let libDir = dir </> "lib64"
                    targetsLibDir = dir </> "targets/x86_64-linux/lib"
                hasLib64 <- test_d libDir
                hasTargetsLib <- test_d targetsLibDir
                when hasLib64 $ copyDir libDir (outputDir </> "lib64")
                when hasTargetsLib $ copyDir targetsLibDir (outputDir </> "lib64")
                -- include - try both direct and targets structure
                let incDir = dir </> "include"
                    targetsIncDir = dir </> "targets/x86_64-linux/include"
                hasInc <- test_d incDir
                hasTargetsInc <- test_d targetsIncDir
                when hasInc $ copyDir incDir (outputDir </> "include")
                when hasTargetsInc $ copyDir targetsIncDir (outputDir </> "include")
                -- nvvm (for nvcc)
                copyDir (dir </> "nvvm") (outputDir </> "nvvm")

        -- Extract system libraries (cuDNN, NCCL, TensorRT)
        let sysLibDir = containerRoot </> "usr/lib/x86_64-linux-gnu"
        hasSysLib <- test_d sysLibDir
        when hasSysLib $ do
            echoErr ":: Extracting cuDNN, NCCL, TensorRT..."
            forM_ targetLibs $ \lib -> do
                -- Find and copy matching libraries
                libs <- findLibs sysLibDir lib
                forM_ libs $ \libPath -> do
                    cp libPath (outputDir </> "lib64" </> filename libPath)

        -- Extract TensorRT-LLM backend if present
        let trtLlmDir = containerRoot </> "opt/tritonserver/backends/tensorrtllm"
        hasTrtLlm <- test_d trtLlmDir
        when hasTrtLlm $ do
            echoErr ":: Extracting TensorRT-LLM backend..."
            mkdirP (outputDir </> "backends")
            copyDir trtLlmDir (outputDir </> "backends/tensorrtllm")

        -- Write version info
        echoErr ":: Writing version info..."
        writeVersionInfo outputDir versions

        -- Fix ELF binaries (RPATHs and interpreter)
        echoErr ":: Patching ELF binaries..."
        patchElf outputDir

        echoErr $ ":: Done! SDK extracted to " <> toTextIgnore outputDir
        echoErr $
            ":: CUDA "
                <> nvCuda versions
                <> ", cuDNN "
                <> nvCudnn versions
                <> ", NCCL "
                <> nvNccl versions

-- | Get version info from container environment variables
getContainerVersions :: Text -> Sh NvidiaVersions
getContainerVersions imageRef = do
    -- Get container config JSON
    configJson <- Crane.config imageRef

    -- Parse JSON and extract Env array
    let envPairs = case decode (BL.fromStrict $ TE.encodeUtf8 configJson) of
            Just (Object obj) -> extractEnv obj
            _ -> []
        lookup' k = fromMaybe "unknown" $ Prelude.lookup k envPairs

    pure
        NvidiaVersions
            { nvCuda = cleanVersion $ lookup' "CUDA_VERSION"
            , nvCudnn = cleanVersion $ lookup' "CUDNN_VERSION"
            , nvNccl = cleanVersion $ lookup' "NCCL_VERSION"
            , nvTensorrt = cleanVersion $ lookup' "TENSORRT_VERSION"
            , nvCutensor = cleanVersion $ lookup' "CUTENSOR_VERSION"
            }
  where
    -- Extract .config.Env array from JSON
    extractEnv :: Object -> [(Text, Text)]
    extractEnv obj = case KM.lookup "config" obj of
        Just (Object cfg) -> case KM.lookup "Env" cfg of
            Just (Array arr) -> mapMaybe parseEnvVar (V.toList arr)
            _ -> []
        _ -> []

    parseEnvVar :: Value -> Maybe (Text, Text)
    parseEnvVar (String t) =
        let (k, rest) = T.breakOn "=" t
         in if T.null rest
                then Nothing
                else Just (k, T.drop 1 rest)
    parseEnvVar _ = Nothing

    -- Remove trailing -1, quotes, etc
    cleanVersion :: Text -> Text
    cleanVersion v =
        let v' = T.replace "\"" "" v
            v'' = if "-1" `T.isSuffixOf` v' then T.dropEnd 2 v' else v'
         in if T.null v'' then "unknown" else v''

{- | Find CUDA directory - looks for versioned dirs like cuda-13.0
Falls back to /usr/local/cuda symlink if available
-}
findCudaDir :: FilePath -> Text -> Sh (Maybe FilePath)
findCudaDir containerRoot cudaVersion = do
    let localDir = containerRoot </> "usr/local"
        -- Extract major.minor from version (e.g., "13.0.1" -> "13.0")
        majorMinor = T.intercalate "." $ Prelude.take 2 $ T.splitOn "." cudaVersion
        versionedDir = localDir </> fromText ("cuda-" <> majorMinor)
        symlinkedDir = localDir </> "cuda"

    hasVersioned <- test_d versionedDir
    hasSymlinked <- test_d symlinkedDir

    pure $
        if hasVersioned
            then Just versionedDir
            else
                if hasSymlinked
                    then Just symlinkedDir
                    else Nothing

-- | Find libraries matching a prefix
findLibs :: FilePath -> Text -> Sh [FilePath]
findLibs dir prefix = do
    exists <- test_d dir
    if exists
        then do
            files <- ls dir
            pure $ filter (hasPrefix prefix . toTextIgnore . filename) files
        else pure []
  where
    hasPrefix p t = T.isPrefixOf p t

-- | Copy directory contents recursively
copyDir :: FilePath -> FilePath -> Sh ()
copyDir src dst = do
    exists <- test_d src
    when exists $ do
        mkdirP dst
        run_ "cp" ["-rL", toTextIgnore src <> "/.", toTextIgnore dst <> "/"]

-- | Write version.json with component versions
writeVersionInfo :: FilePath -> NvidiaVersions -> Sh ()
writeVersionInfo outputDir NvidiaVersions{..} = do
    let versionFile = outputDir </> "version.json"
        json =
            T.unlines
                [ "{"
                , "  \"cuda\": \"" <> nvCuda <> "\","
                , "  \"cudnn\": \"" <> nvCudnn <> "\","
                , "  \"nccl\": \"" <> nvNccl <> "\","
                , "  \"tensorrt\": \"" <> nvTensorrt <> "\","
                , "  \"cutensor\": \"" <> nvCutensor <> "\""
                , "}"
                ]

    liftIO $ TIO.writeFile (T.unpack $ toTextIgnore versionFile) json

{- | Patch all ELF binaries in SDK directory
Uses find to recursively locate all ELF files, then:
  - Sets interpreter for executables
  - Sets RPATH relative to file location
-}
patchElf :: FilePath -> Sh ()
patchElf sdkDir = do
    -- Get absolute SDK root for reliable path comparison
    cwd <- pwd
    let sdkRoot =
            if "/" `T.isPrefixOf` toTextIgnore sdkDir
                then sdkDir
                else cwd </> sdkDir
        sdkRootText = toTextIgnore sdkRoot

    echoErr $ "   Scanning " <> sdkRootText <> " for ELF files..."

    -- Find all regular files (not symlinks)
    allFiles <- findAllFiles sdkRoot

    -- Filter to ELF files and patch each
    patchCount <- M.foldM (patchIfElf sdkRootText) 0 allFiles

    echoErr $ "   Patched " <> pack (show patchCount) <> " ELF files"

-- | Find all regular files recursively (not symlinks)
findAllFiles :: FilePath -> Sh [FilePath]
findAllFiles dir = do
    exists <- test_d dir
    if not exists
        then pure []
        else do
            -- Use find command for recursive search
            output <-
                errExit False $
                    run
                        "find"
                        [ toTextIgnore dir
                        , "-type"
                        , "f"
                        , "-not"
                        , "-name"
                        , "*.py"
                        , "-not"
                        , "-name"
                        , "*.pyc"
                        , "-not"
                        , "-name"
                        , "*.h"
                        , "-not"
                        , "-name"
                        , "*.hpp"
                        , "-not"
                        , "-name"
                        , "*.cuh"
                        , "-not"
                        , "-name"
                        , "*.txt"
                        , "-not"
                        , "-name"
                        , "*.md"
                        , "-not"
                        , "-name"
                        , "*.json"
                        ]
            pure $ map fromText $ filter (not . T.null) $ T.lines output

-- | Patch a file if it's ELF, return 1 if patched, 0 otherwise
patchIfElf :: Text -> Int -> FilePath -> Sh Int
patchIfElf sdkRoot count path = do
    -- Skip symlinks
    isLink <- test_s path
    if isLink
        then pure count
        else do
            -- Check if ELF
            output <- errExit False $ run "file" ["-b", toTextIgnore path]
            if not ("ELF" `T.isInfixOf` output)
                then pure count
                else do
                    -- Determine if executable or shared library
                    let isExe = "executable" `T.isInfixOf` output

                    -- Get the directory containing this file, relative to SDK root
                    let pathText = toTextIgnore path
                        relToRoot = calculateRelPath sdkRoot pathText

                    -- Build RPATH with multiple search paths
                    let rpath =
                            T.intercalate ":" $
                                [ "$ORIGIN"
                                , "$ORIGIN/" <> relToRoot <> "/lib64"
                                , "$ORIGIN/" <> relToRoot <> "/lib"
                                , "$ORIGIN/" <> relToRoot <> "/nvvm/lib64"
                                ]

                    -- Patch RPATH
                    errExit False $ run_ "patchelf" ["--set-rpath", rpath, pathText]

                    -- Set interpreter for executables (not shared objects)
                    when isExe $ do
                        -- Get the dynamic linker path
                        -- For now, use a standard path that works on most Linux
                        errExit False $
                            run_
                                "patchelf"
                                [ "--set-interpreter"
                                , "/lib64/ld-linux-x86-64.so.2"
                                , pathText
                                ]

                    pure (count + 1)

{- | Calculate relative path from file's directory back to SDK root
Takes the SDK root and the absolute file path
e.g., sdkRoot="/tmp/sdk", path="/tmp/sdk/bin/nvcc" -> ".."
      sdkRoot="/tmp/sdk", path="/tmp/sdk/lib64/libfoo.so" -> ".."
      sdkRoot="/tmp/sdk", path="/tmp/sdk/nvvm/lib64/libnvvm.so" -> "../.."
-}
calculateRelPath :: Text -> Text -> Text
calculateRelPath sdkRoot filePath =
    -- Get the file's directory by removing the filename
    let fileDir = T.dropWhileEnd (/= '/') filePath
        -- Remove SDK root prefix to get relative path within SDK
        relPath = fromMaybe fileDir $ T.stripPrefix (sdkRoot <> "/") fileDir
        -- Count directory depth (number of path separators)
        depth = Prelude.length $ filter (== '/') $ T.unpack relPath
     in if depth == 0
            then "."
            else T.intercalate "/" $ Prelude.replicate depth ".."
