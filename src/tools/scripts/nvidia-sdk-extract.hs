{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
nvidia-sdk-extract - Extract NVIDIA SDK components from various sources

This script uses the typed Aleph.Script infrastructure:
  - Aleph.Script.Tools.Crane for OCI image operations
  - Aleph.Script.Tools.Bwrap for FHS sandbox (installer mode)
  - Aleph.Script.Oci for container caching/environment

Modes:
  container   - Extract CUDA/cuDNN/NCCL/TensorRT from NGC container
  tarball     - Extract from downloadable archive
  tritonserver - Full Triton Inference Server extraction
  nccl        - Extract just NCCL from container
  installer   - Run CUDA .run installer in FHS sandbox

The script patches all ELF binaries with relative RPATHs for portability.
-}
module Main where

import Aleph.Script
import qualified Aleph.Script.Oci as Oci
import qualified Aleph.Script.Tools.Bwrap as Bwrap
import qualified Aleph.Script.Tools.Crane as Crane
import Control.Monad (forM_, when)
import qualified Control.Monad as M
import Data.Aeson (Object, Value (..), decode)
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Lazy as BL
import Data.Function ((&))
import Data.Maybe (fromMaybe, mapMaybe)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.IO as TIO
import qualified Data.Vector as V
import System.Environment (getArgs)
import Prelude hiding (FilePath, lines, unlines, unwords, words)

-- ============================================================================
-- Data Types
-- ============================================================================

-- | Version info extracted from container
data NvidiaVersions = NvidiaVersions
    { nvCuda :: Text
    , nvCudnn :: Text
    , nvNccl :: Text
    , nvTensorrt :: Text
    , nvCutensor :: Text
    }
    deriving (Show)

-- | Extraction mode
data ExtractionMode
    = -- | imageRef, outputDir
      ExtractContainer String String
    | -- | url, outputDir, stripComponents
      ExtractTarball String String Int
    | -- | imageRef, outputDir
      ExtractTritonserver String String
    | -- | imageRef, outputDir
      ExtractNccl String String
    | -- | installerPath, outputDir
      ExtractInstaller String String
    | ShowHelp
    deriving (Show)

-- ============================================================================
-- Main
-- ============================================================================

main :: IO ()
main = do
    args <- getArgs
    let mode = parseArgs args
    case mode of
        ShowHelp -> printHelp
        ExtractContainer imageRef outputDir ->
            script $ verbosely $ extractFromContainer (pack imageRef) (fromText $ pack outputDir)
        ExtractTarball url outputDir strip ->
            script $ verbosely $ extractFromTarball (pack url) (fromText $ pack outputDir) strip
        ExtractTritonserver imageRef outputDir ->
            script $ verbosely $ extractTritonserver (pack imageRef) (fromText $ pack outputDir)
        ExtractNccl imageRef outputDir ->
            script $ verbosely $ extractNccl (pack imageRef) (fromText $ pack outputDir)
        ExtractInstaller installerPath outputDir ->
            script $ verbosely $ extractFromInstaller (pack installerPath) (fromText $ pack outputDir)

parseArgs :: [String] -> ExtractionMode
parseArgs = \case
    ["container", imageRef, outputDir] -> ExtractContainer imageRef outputDir
    ["tarball", url, outputDir] -> ExtractTarball url outputDir 0
    ["tarball", url, outputDir, "--strip", n] -> ExtractTarball url outputDir (read n)
    ["tritonserver", imageRef, outputDir] -> ExtractTritonserver imageRef outputDir
    ["nccl", imageRef, outputDir] -> ExtractNccl imageRef outputDir
    ["installer", path, outputDir] -> ExtractInstaller path outputDir
    _ -> ShowHelp

printHelp :: IO ()
printHelp = do
    putStrLn "nvidia-sdk-extract - Extract NVIDIA SDK components"
    putStrLn ""
    putStrLn "Usage:"
    putStrLn "  nvidia-sdk-extract container <image-ref> <output-dir>"
    putStrLn "    Extract CUDA, cuDNN, NCCL, TensorRT from NGC container"
    putStrLn ""
    putStrLn "  nvidia-sdk-extract tarball <url> <output-dir> [--strip N]"
    putStrLn "    Extract from tarball (cuDNN, TensorRT, cuTensor)"
    putStrLn ""
    putStrLn "  nvidia-sdk-extract tritonserver <image-ref> <output-dir>"
    putStrLn "    Extract Triton Inference Server from NGC container"
    putStrLn ""
    putStrLn "  nvidia-sdk-extract nccl <image-ref> <output-dir>"
    putStrLn "    Extract NCCL libraries from NGC container"
    putStrLn ""
    putStrLn "  nvidia-sdk-extract installer <cuda.run> <output-dir>"
    putStrLn "    Run CUDA .run installer in FHS sandbox"
    putStrLn ""
    putStrLn "Examples:"
    putStrLn "  nvidia-sdk-extract container nvcr.io/nvidia/tritonserver:25.11-py3 ./sdk"
    putStrLn "  nvidia-sdk-extract tritonserver nvcr.io/nvidia/tritonserver:25.11-py3 ./triton"
    putStrLn "  nvidia-sdk-extract installer cuda_13.0.2_580.95.05_linux.run ./cuda"

-- ============================================================================
-- Container Extraction (CUDA, cuDNN, NCCL, TensorRT)
-- ============================================================================

-- | Target libraries to extract from container
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
    , "libnvjitlink"
    , "libcupti"
    ]

extractFromContainer :: Text -> FilePath -> Sh ()
extractFromContainer imageRef outputDir = do
    echoErr $ ":: Extracting NVIDIA SDK from " <> imageRef

    -- Get version info from container config using typed Crane wrapper
    echoErr ":: Reading container config..."
    versions <- getContainerVersions imageRef

    -- Use Oci module for pulling/caching
    let ociConfig = Oci.defaultConfig
    rootfs <- Oci.pullOrCache ociConfig imageRef

    -- Create output structure
    echoErr ":: Creating SDK layout..."
    mkdirP (outputDir </> "bin")
    mkdirP (outputDir </> "lib64")
    mkdirP (outputDir </> "include")
    mkdirP (outputDir </> "nvvm")

    -- Extract CUDA toolkit
    cudaDir <- findCudaDir rootfs (nvCuda versions)
    case cudaDir of
        Nothing -> echoErr ":: Warning: CUDA directory not found"
        Just dir -> do
            echoErr $ ":: Extracting CUDA toolkit " <> nvCuda versions <> "..."
            extractCudaToolkit dir outputDir

    -- Extract system libraries (cuDNN, NCCL, TensorRT)
    extractSystemLibs rootfs outputDir

    -- Extract TensorRT-LLM backend if present
    extractTrtLlmBackend rootfs outputDir

    -- Write version info
    writeVersionInfo outputDir versions

    -- Fix ELF binaries
    echoErr ":: Patching ELF binaries..."
    patchElf outputDir

    echoErr $ ":: Done! SDK extracted to " <> toTextIgnore outputDir
    echoErr $ ":: CUDA " <> nvCuda versions <> ", cuDNN " <> nvCudnn versions

-- | Extract CUDA toolkit from container
extractCudaToolkit :: FilePath -> FilePath -> Sh ()
extractCudaToolkit cudaDir outputDir = do
    -- bin (nvcc, etc)
    let binDir = cudaDir </> "bin"
    hasBin <- test_d binDir
    when hasBin $ do
        files <- ls binDir
        forM_ files $ \f -> do
            isFile <- test_f f
            when isFile $ errExit False $ cp f (outputDir </> "bin" </> filename f)

    -- lib64 - try both direct and targets structure
    copyDirContents (cudaDir </> "lib64") (outputDir </> "lib64")
    copyDirContents (cudaDir </> "targets/x86_64-linux/lib") (outputDir </> "lib64")
    copyDirContents (cudaDir </> "targets/sbsa-linux/lib") (outputDir </> "lib64")

    -- include
    copyDirContents (cudaDir </> "include") (outputDir </> "include")
    copyDirContents (cudaDir </> "targets/x86_64-linux/include") (outputDir </> "include")
    copyDirContents (cudaDir </> "targets/sbsa-linux/include") (outputDir </> "include")

    -- nvvm
    copyDirContents (cudaDir </> "nvvm") (outputDir </> "nvvm")

-- | Extract system libraries
extractSystemLibs :: FilePath -> FilePath -> Sh ()
extractSystemLibs containerRoot outputDir = do
    let sysLibDirs =
            [ containerRoot </> "usr/lib/x86_64-linux-gnu"
            , containerRoot </> "usr/lib/aarch64-linux-gnu"
            , containerRoot </> "usr/local/lib"
            ]

    echoErr ":: Extracting cuDNN, NCCL, TensorRT..."
    forM_ sysLibDirs $ \sysLibDir -> do
        exists <- test_d sysLibDir
        when exists $ do
            forM_ targetLibs $ \prefix -> do
                libs <- findLibsWithPrefix sysLibDir prefix
                forM_ libs $ \libPath -> do
                    errExit False $ run_ "cp" ["-L", "--no-preserve=mode", toTextIgnore libPath, toTextIgnore (outputDir </> "lib64") <> "/"]

-- | Extract TensorRT-LLM backend if present
extractTrtLlmBackend :: FilePath -> FilePath -> Sh ()
extractTrtLlmBackend containerRoot outputDir = do
    let trtLlmDir = containerRoot </> "opt/tritonserver/backends/tensorrtllm"
    hasTrtLlm <- test_d trtLlmDir
    when hasTrtLlm $ do
        echoErr ":: Extracting TensorRT-LLM backend..."
        mkdirP (outputDir </> "backends")
        copyDir trtLlmDir (outputDir </> "backends/tensorrtllm")

-- ============================================================================
-- Tarball Extraction
-- ============================================================================

extractFromTarball :: Text -> FilePath -> Int -> Sh ()
extractFromTarball url outputDir stripComponents = do
    echoErr $ ":: Extracting from tarball: " <> url

    withTmpDir $ \tmpDir -> do
        let tarball = tmpDir </> "archive"

        -- Download
        echoErr ":: Downloading..."
        run_ "curl" ["-fsSL", "-o", toTextIgnore tarball, url]

        -- Detect compression and extract
        echoErr ":: Extracting..."
        mkdirP outputDir
        let ext = T.toLower $ T.takeWhileEnd (/= '.') url
            tarFlags = case ext of
                "gz" -> ["-xzf"]
                "xz" -> ["-xJf"]
                "bz2" -> ["-xjf"]
                "tar" -> ["-xf"]
                _ -> ["-xf"] -- let tar auto-detect
            stripArg =
                if stripComponents > 0
                    then ["--strip-components=" <> pack (show stripComponents)]
                    else []
        run_ "tar" (tarFlags ++ stripArg ++ [toTextIgnore tarball, "-C", toTextIgnore outputDir])

        -- Create lib symlink if needed
        hasLib64 <- test_d (outputDir </> "lib64")
        hasLib <- test_e (outputDir </> "lib")
        when (hasLib64 && not hasLib) $ do
            symlink "lib64" (outputDir </> "lib")

        -- Patch ELF binaries
        echoErr ":: Patching ELF binaries..."
        patchElf outputDir

        echoErr $ ":: Done! Extracted to " <> toTextIgnore outputDir

-- ============================================================================
-- Installer Extraction (FHS Sandbox via Bwrap)
-- ============================================================================

-- | Run CUDA .run installer in an FHS sandbox using typed Bwrap
extractFromInstaller :: Text -> FilePath -> Sh ()
extractFromInstaller installerPath outputDir = do
    echoErr $ ":: Running installer in FHS sandbox: " <> installerPath

    -- Check installer exists
    installerExists <- test_f (unpack installerPath)
    unless installerExists $ do
        die $ "Installer not found: " <> installerPath

    -- Make output directory
    mkdirP outputDir
    absOutputDir <- realpath outputDir

    -- Build FHS sandbox with minimal dependencies
    -- The installer needs: coreutils, perl, which, file
    -- We use symlinkBind to create symlinks inside the sandbox
    let sandbox =
            Bwrap.defaults
                -- Mount /nix/store read-only for tools
                & Bwrap.roBind "/nix/store" "/nix/store"
                -- Core FHS structure
                & Bwrap.dev "/dev"
                & Bwrap.proc "/proc"
                & Bwrap.tmpfs "/tmp"
                & Bwrap.tmpfs "/run"
                -- Bind installer and output (read-write)
                & Bwrap.roBind (unpack installerPath) "/installer.run"
                & Bwrap.bind absOutputDir "/output"
                -- FHS paths that installer expects
                & Bwrap.dir "/usr"
                & Bwrap.dir "/usr/bin"
                & Bwrap.dir "/usr/lib"
                & Bwrap.dir "/usr/local"
                & Bwrap.dir "/lib64"
                -- Environment
                & Bwrap.setenv "PATH" "/nix/store/bin:/usr/bin:/bin"
                & Bwrap.setenv "HOME" "/tmp"
                & Bwrap.chdir "/tmp"
                & Bwrap.dieWithParent
                & Bwrap.unsharePid

    echoErr ":: Extracting via FHS sandbox..."
    -- Run the installer with --silent --toolkit --toolkitpath
    Bwrap.bwrap_
        sandbox
        [ "sh"
        , "/installer.run"
        , "--silent"
        , "--toolkit"
        , "--toolkitpath=/output"
        , "--no-opengl-libs"
        , "--override"
        ]

    -- Create lib symlink
    hasLib64 <- test_d (outputDir </> "lib64")
    hasLib <- test_e (outputDir </> "lib")
    when (hasLib64 && not hasLib) $ do
        symlink "lib64" (outputDir </> "lib")

    -- Move libcuda.so to stubs (it's a driver stub, not the real driver)
    let stubsDir = outputDir </> "lib64/stubs"
    mkdirP stubsDir
    hasCuda <- test_f (outputDir </> "lib64/libcuda.so")
    when hasCuda $ do
        mv (outputDir </> "lib64/libcuda.so") (stubsDir </> "libcuda.so")

    -- Patch ELF binaries
    echoErr ":: Patching ELF binaries..."
    patchElf outputDir

    echoErr $ ":: Done! CUDA toolkit extracted to " <> toTextIgnore outputDir

-- ============================================================================
-- Tritonserver Extraction
-- ============================================================================

extractTritonserver :: Text -> FilePath -> Sh ()
extractTritonserver imageRef outputDir = do
    echoErr $ ":: Extracting Tritonserver from " <> imageRef

    -- Use Oci module for pulling/caching
    let ociConfig = Oci.defaultConfig
    rootfs <- Oci.pullOrCache ociConfig imageRef

    -- Create output structure
    mkdirP (outputDir </> "bin")
    mkdirP (outputDir </> "lib")
    mkdirP (outputDir </> "include")
    mkdirP (outputDir </> "backends")
    mkdirP (outputDir </> "python")

    -- Copy tritonserver installation
    let tritonDir = rootfs </> "opt/tritonserver"
    hasTriton <- test_d tritonDir
    when hasTriton $ do
        echoErr ":: Copying tritonserver..."
        copyDir tritonDir outputDir

    -- Copy TensorRT-LLM if present
    let trtLlmDir = rootfs </> "opt/tensorrt_llm"
    hasTrtLlm <- test_d trtLlmDir
    when hasTrtLlm $ do
        echoErr ":: Copying TensorRT-LLM..."
        mkdirP (outputDir </> "tensorrt_llm")
        copyDir trtLlmDir (outputDir </> "tensorrt_llm")

    -- Copy additional libraries from container
    echoErr ":: Copying container libraries..."
    copyContainerLibs rootfs outputDir

    -- Copy Python packages
    echoErr ":: Copying Python packages..."
    copyPythonPackages rootfs (outputDir </> "python")

    -- Create lib64 -> lib symlink
    hasLib <- test_d (outputDir </> "lib")
    hasLib64 <- test_e (outputDir </> "lib64")
    when (hasLib && not hasLib64) $ do
        symlink "lib" (outputDir </> "lib64")

    echoErr $ ":: Done! Tritonserver extracted to " <> toTextIgnore outputDir

-- | Copy additional libraries from container that tritonserver needs
copyContainerLibs :: FilePath -> FilePath -> Sh ()
copyContainerLibs containerRoot outputDir = do
    -- Libraries that aren't in nixpkgs or have version mismatches
    let specialLibs =
            [ "libcupti"
            , "libb64"
            , "libdcgm"
            , "libcusparseLt"
            , "libnvshmem"
            , "libcaffe2_nvrtc"
            , "libicu"
            ]

    let searchDirs =
            [ containerRoot </> "usr/lib/x86_64-linux-gnu"
            , containerRoot </> "usr/local/lib"
            , containerRoot </> "opt/tritonserver/lib"
            ]

    forM_ searchDirs $ \searchDir -> do
        exists <- test_d searchDir
        when exists $ do
            forM_ specialLibs $ \prefix -> do
                libs <- findLibsWithPrefix searchDir prefix
                forM_ libs $ \libPath -> do
                    let destPath = outputDir </> "lib" </> filename libPath
                    errExit False $ run_ "cp" ["-L", "--no-preserve=mode", toTextIgnore libPath, toTextIgnore destPath]

-- | Copy Python packages from container
copyPythonPackages :: FilePath -> FilePath -> Sh ()
copyPythonPackages containerRoot outputDir = do
    let pyDirs =
            [ containerRoot </> "usr/lib/python3/dist-packages"
            , containerRoot </> "usr/local/lib/python3.12/dist-packages"
            , containerRoot </> "opt/tritonserver/python"
            ]

    forM_ pyDirs $ \pyDir -> do
        exists <- test_d pyDir
        when exists $ do
            copyDirContents pyDir outputDir

-- ============================================================================
-- NCCL Extraction
-- ============================================================================

extractNccl :: Text -> FilePath -> Sh ()
extractNccl imageRef outputDir = do
    echoErr $ ":: Extracting NCCL from " <> imageRef

    -- Use Oci module for pulling/caching
    let ociConfig = Oci.defaultConfig
    rootfs <- Oci.pullOrCache ociConfig imageRef

    -- Create output structure
    mkdirP (outputDir </> "lib")
    mkdirP (outputDir </> "include")

    -- Find and copy NCCL libraries
    let searchDirs =
            [ rootfs </> "usr/lib/x86_64-linux-gnu"
            , rootfs </> "usr/local/lib"
            ]

    forM_ searchDirs $ \searchDir -> do
        exists <- test_d searchDir
        when exists $ do
            libs <- findLibsWithPrefix searchDir "libnccl"
            forM_ libs $ \libPath -> do
                errExit False $ run_ "cp" ["-L", "--no-preserve=mode", toTextIgnore libPath, toTextIgnore (outputDir </> "lib") <> "/"]

    -- Find and copy NCCL headers
    let incDirs =
            [ rootfs </> "usr/include"
            , rootfs </> "usr/local/include"
            ]

    forM_ incDirs $ \incDir -> do
        exists <- test_d incDir
        when exists $ do
            headers <- findFilesWithPrefix incDir "nccl"
            forM_ headers $ \h -> do
                errExit False $ run_ "cp" ["-L", "--no-preserve=mode", toTextIgnore h, toTextIgnore (outputDir </> "include") <> "/"]

    -- Create lib64 -> lib symlink
    symlink "lib" (outputDir </> "lib64")

    -- Patch ELF
    patchElf outputDir

    echoErr $ ":: Done! NCCL extracted to " <> toTextIgnore outputDir

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- | Get version info from container config using typed Crane wrapper
getContainerVersions :: Text -> Sh NvidiaVersions
getContainerVersions imageRef = do
    configJson <- Crane.config imageRef

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
    extractEnv :: Object -> [(Text, Text)]
    extractEnv obj = case KM.lookup "config" obj of
        Just (Object cfg) -> case KM.lookup "Env" cfg of
            Just (Array arr) -> mapMaybe parseEnvVar (V.toList arr)
            _ -> []
        _ -> []

    parseEnvVar :: Value -> Maybe (Text, Text)
    parseEnvVar (String t) =
        let (k, rest) = T.breakOn "=" t
         in if T.null rest then Nothing else Just (k, T.drop 1 rest)
    parseEnvVar _ = Nothing

    cleanVersion :: Text -> Text
    cleanVersion v =
        let v' = T.replace "\"" "" v
            v'' = if "-1" `T.isSuffixOf` v' then T.dropEnd 2 v' else v'
         in if T.null v'' then "unknown" else v''

-- | Find CUDA directory
findCudaDir :: FilePath -> Text -> Sh (Maybe FilePath)
findCudaDir containerRoot cudaVersion = do
    let localDir = containerRoot </> "usr/local"
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

-- | Find libraries with a given prefix
findLibsWithPrefix :: FilePath -> Text -> Sh [FilePath]
findLibsWithPrefix dir prefix = do
    exists <- test_d dir
    if not exists
        then pure []
        else do
            files <- ls dir
            pure $ filter (hasPrefix prefix . toTextIgnore . filename) files
  where
    hasPrefix p t = T.isPrefixOf p t

-- | Find files (including headers) with a given prefix
findFilesWithPrefix :: FilePath -> Text -> Sh [FilePath]
findFilesWithPrefix dir prefix = do
    exists <- test_d dir
    if not exists
        then pure []
        else do
            files <- ls dir
            pure $ filter (hasPrefix prefix . toTextIgnore . filename) files
  where
    hasPrefix p t = T.isPrefixOf p t

-- | Copy directory contents (not the directory itself)
copyDirContents :: FilePath -> FilePath -> Sh ()
copyDirContents src dst = do
    exists <- test_d src
    when exists $ do
        mkdirP dst
        errExit False $ run_ "cp" ["-rL", "--no-preserve=mode", toTextIgnore src <> "/.", toTextIgnore dst <> "/"]

-- | Copy directory recursively
copyDir :: FilePath -> FilePath -> Sh ()
copyDir src dst = do
    exists <- test_d src
    when exists $ do
        mkdirP dst
        errExit False $ run_ "cp" ["-rL", "--no-preserve=mode", toTextIgnore src <> "/.", toTextIgnore dst <> "/"]

-- | Write version.json
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

-- | Patch all ELF binaries with relative RPATHs
patchElf :: FilePath -> Sh ()
patchElf sdkDir = do
    cwd <- pwd
    let sdkRoot =
            if "/" `T.isPrefixOf` toTextIgnore sdkDir
                then sdkDir
                else cwd </> sdkDir
        sdkRootText = toTextIgnore sdkRoot

    allFiles <- findAllFiles sdkRoot
    patchCount <- M.foldM (patchIfElf sdkRootText) 0 allFiles
    echoErr $ "   Patched " <> pack (show patchCount) <> " ELF files"

findAllFiles :: FilePath -> Sh [FilePath]
findAllFiles dir = do
    exists <- test_d dir
    if not exists
        then pure []
        else do
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

patchIfElf :: Text -> Int -> FilePath -> Sh Int
patchIfElf sdkRoot count path = do
    isLink <- test_s path
    if isLink
        then pure count
        else do
            output <- errExit False $ run "file" ["-b", toTextIgnore path]
            if not ("ELF" `T.isInfixOf` output)
                then pure count
                else do
                    let isExe = "executable" `T.isInfixOf` output
                        pathText = toTextIgnore path
                        relToRoot = calculateRelPath sdkRoot pathText
                        rpath =
                            T.intercalate
                                ":"
                                [ "$ORIGIN"
                                , "$ORIGIN/" <> relToRoot <> "/lib64"
                                , "$ORIGIN/" <> relToRoot <> "/lib"
                                , "$ORIGIN/" <> relToRoot <> "/nvvm/lib64"
                                ]

                    errExit False $ run_ "patchelf" ["--set-rpath", rpath, pathText]

                    when isExe $ do
                        errExit False $
                            run_
                                "patchelf"
                                ["--set-interpreter", "/lib64/ld-linux-x86-64.so.2", pathText]

                    pure (count + 1)

calculateRelPath :: Text -> Text -> Text
calculateRelPath sdkRoot filePath =
    let fileDir = T.dropWhileEnd (/= '/') filePath
        relPath = fromMaybe fileDir $ T.stripPrefix (sdkRoot <> "/") fileDir
        depth = Prelude.length $ filter (== '/') $ T.unpack relPath
     in if depth == 0
            then "."
            else T.intercalate "/" $ Prelude.replicate depth ".."

-- | Get real path, needed for Bwrap bind mounts
realpath :: FilePath -> Sh FilePath
realpath p = fromText . strip <$> run "realpath" [toTextIgnore p]
