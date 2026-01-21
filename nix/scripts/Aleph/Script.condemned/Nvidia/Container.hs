{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Nvidia.Container
Description : Extract NVIDIA SDK from NGC containers

Extracts CUDA toolkit, cuDNN, NCCL, TensorRT from NGC containers.
Used for components not available as PyPI wheels (e.g., nvcc, cuda-gdb).

== Architecture

Container extraction is a two-phase process:

1. Pull: Download and unpack container image (crane export)
2. Extract: Copy relevant files to output directory, patch ELF

== Usage

@
import Aleph.Script
import qualified Aleph.Script.Nvidia.Container as Container
import qualified Aleph.Script.Nvidia.Versions as V

main = script $ do
  -- Extract from tritonserver container
  let ref = V.tritonserver V.currentPlatform
  Container.extractSdk (V.containerRef ref) "/tmp/nvidia-sdk" Container.CudaToolkit
@
-}
module Aleph.Script.Nvidia.Container (
    -- * Extraction modes
    ExtractMode (..),

    -- * Operations
    pullImage,
    extractSdk,
    extractTritonserver,

    -- * Low-level
    findCudaDir,
    copyLibraries,
    copyCudaLibraries,
    createLibrarySymlinks,
    patchElfBinaries,
) where

import Aleph.Script hiding (FilePath)
import Control.Monad (forM_, when)
import qualified Data.List as L
import qualified Data.Text as T

-- ============================================================================
-- Library Prefixes (centralized)
-- ============================================================================

-- | Core CUDA runtime libraries
cudaRuntimeLibs :: [Text]
cudaRuntimeLibs =
    [ "libcudart"
    , "libcublas"
    , "libcublasLt"
    , "libcufft"
    , "libcurand"
    , "libcusolver"
    , "libcusparse"
    , "libnvrtc"
    , "libnvJitLink" -- Note: camelCase
    ]

-- | CUDA libraries from system paths (cuDNN, TensorRT, etc.)
cudaSystemLibs :: [Text]
cudaSystemLibs =
    [ "libcudnn"
    , "libnvinfer"
    , "libnvonnxparser"
    , "libnccl"
    , "libcutensor"
    ]

-- | Extra libraries needed by tritonserver
tritonExtraLibs :: [Text]
tritonExtraLibs =
    [ "libcupti"
    , "libb64"
    , "libdcgm"
    , "libcusparseLt"
    , "libnvshmem"
    , "libcaffe2_nvrtc"
    , "libcufile"
    , "libOpenCL"
    ]

-- ============================================================================
-- Types
-- ============================================================================

-- | What to extract from the container
data ExtractMode
    = -- | CUDA toolkit (nvcc, cuda-gdb, libraries)
      CudaToolkit
    | -- | Just runtime libraries (no compiler)
      CudaRuntime
    | -- | Triton Inference Server + TensorRT-LLM
      Tritonserver
    deriving (Show, Eq)

-- ============================================================================
-- Container Operations
-- ============================================================================

-- | Pull a container image to a directory
pullImage :: Text -> FilePath -> Sh ()
pullImage imageRef outputDir = do
    echoErr $ ":: Pulling " <> imageRef
    mkdirP outputDir
    -- crane export outputs a tarball, pipe to tar
    run_ "sh" ["-c", "crane export " <> imageRef <> " - | tar -xf - -C " <> toTextIgnore outputDir]
    echoErr $ ":: Extracted to " <> toTextIgnore outputDir

{- | Extract NVIDIA SDK from a container rootfs
Note: Does NOT patch ELF binaries - that's handled by autoPatchelfHook in Nix
-}
extractSdk :: FilePath -> FilePath -> ExtractMode -> Sh ()
extractSdk rootfs outputDir mode = do
    echoErr $ ":: Extracting SDK (" <> pack (show mode) <> ")"
    mkdirP outputDir

    case mode of
        CudaToolkit -> extractCudaToolkit rootfs outputDir
        CudaRuntime -> extractCudaRuntime rootfs outputDir
        Tritonserver -> extractTritonserver rootfs outputDir

    -- Make all files writable (needed for autoPatchelf later)
    errExit False $ run_ "chmod" ["-R", "u+w", toTextIgnore outputDir]

    -- Make executables in bin/ executable
    let binDir = outputDir </> "bin"
    hasBin <- test_d binDir
    when hasBin $ do
        echoErr ":: Setting executable permissions on bin/"
        errExit False $ run_ "sh" ["-c", "chmod +x " <> toTextIgnore binDir <> "/*"]

    echoErr $ ":: SDK extracted to " <> toTextIgnore outputDir

-- ============================================================================
-- CUDA Toolkit Extraction
-- ============================================================================

-- | Extract full CUDA toolkit (nvcc, libraries, headers)
extractCudaToolkit :: FilePath -> FilePath -> Sh ()
extractCudaToolkit rootfs outputDir = do
    -- Create output directories
    forM_ ["bin", "lib64", "include", "nvvm"] $ \d ->
        mkdirP (outputDir </> d)

    -- Find CUDA directory
    mCudaDir <- findCudaDir rootfs
    case mCudaDir of
        Nothing -> echoErr ":: Warning: No CUDA directory found"
        Just cd -> do
            echoErr $ ":: Found CUDA at " <> toTextIgnore cd

            -- Copy bin (nvcc, cuda-gdb, etc)
            whenM (test_d (cd </> "bin")) $ do
                echoErr ":: Copying bin/"
                copyDirContents (cd </> "bin") (outputDir </> "bin")

            -- Copy lib64 (try multiple locations)
            let libDirs =
                    [ cd </> "lib64"
                    , cd </> "targets/x86_64-linux/lib"
                    , cd </> "targets/sbsa-linux/lib"
                    ]
            forM_ libDirs $ \d -> copyLibDir d (outputDir </> "lib64")

            -- Copy include
            let includeDirs =
                    [ cd </> "include"
                    , cd </> "targets/x86_64-linux/include"
                    , cd </> "targets/sbsa-linux/include"
                    ]
            forM_ includeDirs $ \d -> copyIncludeDir d (outputDir </> "include")

            -- Copy nvvm (for nvcc)
            whenM (test_d (cd </> "nvvm")) $ do
                echoErr ":: Copying nvvm/"
                copyDirContents (cd </> "nvvm") (outputDir </> "nvvm")

    -- Copy system libraries (cuDNN, NCCL, TensorRT from /usr/lib)
    copySystemLibraries rootfs outputDir

    -- Create lib -> lib64 symlink
    unlessM (test_e (outputDir </> "lib")) $
        symlink "lib64" (outputDir </> "lib")

-- | Extract just CUDA runtime libraries (no compiler)
extractCudaRuntime :: FilePath -> FilePath -> Sh ()
extractCudaRuntime rootfs outputDir = do
    mkdirP (outputDir </> "lib64")
    mkdirP (outputDir </> "include")

    -- Find CUDA directory
    mCudaDir <- findCudaDir rootfs
    forM_ mCudaDir $ \cd -> do
        -- Copy only runtime libraries
        copyLibraries (cd </> "lib64") (outputDir </> "lib64") cudaRuntimeLibs
        -- Copy headers
        copyIncludeDir (cd </> "include") (outputDir </> "include")

    -- Copy system libraries
    copySystemLibraries rootfs outputDir

    -- Create lib symlink
    unlessM (test_e (outputDir </> "lib")) $
        symlink "lib64" (outputDir </> "lib")

-- ============================================================================
-- Tritonserver Extraction
-- ============================================================================

-- | Extract Triton Inference Server
extractTritonserver :: FilePath -> FilePath -> Sh ()
extractTritonserver rootfs outputDir = do
    echoErr ":: Extracting Triton Inference Server"

    -- Create output directories
    forM_ ["bin", "lib", "include", "backends", "python"] $ \d ->
        mkdirP (outputDir </> d)

    -- Copy tritonserver
    let tritonDir = rootfs </> "opt/tritonserver"
    whenM (test_d tritonDir) $ do
        echoErr ":: Copying tritonserver"
        copyDirContents tritonDir outputDir

    -- Copy TensorRT-LLM
    let trtllmDir = rootfs </> "opt/tensorrt_llm"
    whenM (test_d trtllmDir) $ do
        echoErr ":: Copying tensorrt_llm"
        mkdirP (outputDir </> "tensorrt_llm")
        copyDirContents trtllmDir (outputDir </> "tensorrt_llm")

    -- Copy CUDA libraries from container (needed by backends)
    copyCudaLibraries rootfs (outputDir </> "lib")

    -- Copy extra libraries needed by tritonserver
    copyExtraLibs rootfs (outputDir </> "lib") tritonExtraLibs

    -- Copy ICU libraries (for protobuf, version 74)
    copyExtraLibs rootfs (outputDir </> "lib") ["libicu"]

    -- Copy Python packages
    let pyPaths =
            [ "usr/lib/python3/dist-packages"
            , "usr/local/lib/python3.12/dist-packages"
            ]
    forM_ pyPaths $ \pyPath -> do
        let pyDir = rootfs </> fromText pyPath
        whenM (test_d pyDir) $
            copyDirContents pyDir (outputDir </> "python")

    -- Create lib64 -> lib symlink if missing
    unlessM (test_e (outputDir </> "lib64")) $ do
        echoErr ":: Creating lib64 -> lib symlink"
        symlink "lib" (outputDir </> "lib64")

    -- Create missing .so symlinks for versioned libraries
    createLibrarySymlinks (outputDir </> "lib")

-- | Copy extra libraries by searching recursively (only .so files)
copyExtraLibs :: FilePath -> FilePath -> [Text] -> Sh ()
copyExtraLibs rootfs dstDir prefixes =
    forM_ prefixes $ \prefix -> do
        files <- findFilesRecursive rootfs (prefix <> "*.so*")
        forM_ files $ \f -> copyFile f dstDir

-- | when for monadic predicates
whenM :: (Monad m) => m Bool -> m () -> m ()
whenM cond action = cond >>= \b -> when b action

-- | Copy CUDA runtime libraries from container
copyCudaLibraries :: FilePath -> FilePath -> Sh ()
copyCudaLibraries rootfs outputLib = do
    echoErr ":: Copying CUDA libraries from container"

    -- Copy from CUDA toolkit (libcudart, libcublas, etc.)
    mCudaDir <- findCudaDir rootfs
    forM_ mCudaDir $ \cd -> do
        -- Try target-specific lib paths first, then lib64
        let cudaLibDirs =
                [ cd </> "targets/x86_64-linux/lib"
                , cd </> "targets/sbsa-linux/lib"
                , cd </> "lib64"
                ]
        forM_ cudaLibDirs $ \libDir ->
            copyLibraries libDir outputLib cudaRuntimeLibs

    -- Copy from system paths (cuDNN, TensorRT, NCCL)
    mLibDir <- findSystemLibDir rootfs
    forM_ mLibDir $ \libDir ->
        copyLibraries libDir outputLib cudaSystemLibs

-- ============================================================================
-- Helper Functions
-- ============================================================================

{- | Find CUDA directory in rootfs (handles versioned paths)
Prefers actual directories over symlinks (cuda-13.0 over cuda-13)
-}
findCudaDir :: FilePath -> Sh (Maybe FilePath)
findCudaDir rootfs = do
    -- Try versioned paths (cuda-13.0, cuda-12.4, etc) - prefer longer version strings
    -- Use find to get actual directories, not symlinks
    let cudaBase = rootfs </> "usr/local"
    output <- errExit False $ run "find" [toTextIgnore cudaBase, "-maxdepth", "1", "-type", "d", "-name", "cuda-*"]
    let dirs = filter (not . T.null) $ T.lines output
        -- Sort by length descending to prefer cuda-13.0 over cuda-13
        sortedDirs = reverse $ L.sortOn T.length dirs
    case sortedDirs of
        (d : _) -> pure $ Just (fromText d)
        [] -> do
            -- Try /usr/local/cuda (may be a symlink)
            let cudaPath = rootfs </> "usr/local/cuda"
            exists <- test_d cudaPath
            pure $ if exists then Just cudaPath else Nothing

-- | Find system library directory (x86_64 or aarch64)
findSystemLibDir :: FilePath -> Sh (Maybe FilePath)
findSystemLibDir rootfs = do
    let x86Dir = rootfs </> "usr/lib/x86_64-linux-gnu"
        armDir = rootfs </> "usr/lib/aarch64-linux-gnu"
    hasX86 <- test_d x86Dir
    if hasX86
        then pure (Just x86Dir)
        else do
            hasArm <- test_d armDir
            pure $ if hasArm then Just armDir else Nothing

-- | Find directories matching a glob pattern
findDirs :: FilePath -> Sh [FilePath]
findDirs globPat = do
    output <- errExit False $ run "sh" ["-c", "ls -d " <> toTextIgnore globPat <> " 2>/dev/null"]
    pure $ map fromText $ filter (not . T.null) $ T.lines output

-- | Find files matching a pattern in a directory (includes symlinks)
findFiles :: FilePath -> Text -> Sh [FilePath]
findFiles dir namePat = do
    exists <- test_d dir
    if exists
        then do
            -- Use -L to follow symlinks, find both files and symlinks
            output <- errExit False $ run "find" ["-L", toTextIgnore dir, "-maxdepth", "1", "-name", namePat, "-type", "f"]
            pure $ map fromText $ filter (not . T.null) $ T.lines output
        else pure []

-- | Find files recursively matching a pattern (includes symlinks)
findFilesRecursive :: FilePath -> Text -> Sh [FilePath]
findFilesRecursive dir namePat = do
    output <- errExit False $ run "find" ["-L", toTextIgnore dir, "-name", namePat, "-type", "f"]
    pure $ map fromText $ filter (not . T.null) $ T.lines output

-- | Copy directory contents (cp -rL --no-preserve=mode src/. dst/)
copyDirContents :: FilePath -> FilePath -> Sh ()
copyDirContents src dst = do
    exists <- test_d src
    when exists $
        errExit False $
            run_ "cp" ["-rL", "--no-preserve=mode", toTextIgnore src <> "/.", toTextIgnore dst <> "/"]

-- | Copy a single file
copyFile :: FilePath -> FilePath -> Sh ()
copyFile src dst =
    errExit False $ run_ "cp" ["-L", "--no-preserve=mode", toTextIgnore src, toTextIgnore dst <> "/"]

-- | Copy a library directory if it exists
copyLibDir :: FilePath -> FilePath -> Sh ()
copyLibDir src dst = do
    exists <- test_d src
    when exists $ do
        echoErr $ ":: Copying " <> toTextIgnore src
        copyDirContents src dst

-- | Copy an include directory if it exists
copyIncludeDir :: FilePath -> FilePath -> Sh ()
copyIncludeDir src dst = do
    exists <- test_d src
    when exists $ do
        echoErr $ ":: Copying headers from " <> toTextIgnore src
        copyDirContents src dst

-- | Copy libraries matching a prefix
copyMatchingLibs :: FilePath -> FilePath -> Text -> Sh ()
copyMatchingLibs srcDir dstDir prefix = do
    files <- findFiles srcDir (prefix <> "*")
    forM_ files $ \f -> copyFile f dstDir

-- | Copy system libraries (cuDNN, NCCL, TensorRT from /usr/lib)
copySystemLibraries :: FilePath -> FilePath -> Sh ()
copySystemLibraries rootfs outputDir = do
    mLibDir <- findSystemLibDir rootfs
    case mLibDir of
        Nothing -> echoErr ":: Warning: No system lib directory found"
        Just libDir -> do
            echoErr $ ":: Copying system libraries from " <> toTextIgnore libDir
            let allLibs = cudaRuntimeLibs <> cudaSystemLibs
            forM_ allLibs $ \prefix ->
                copyMatchingLibs libDir (outputDir </> "lib64") prefix

-- | Copy libraries from a directory
copyLibraries :: FilePath -> FilePath -> [Text] -> Sh ()
copyLibraries srcDir dstDir prefixes =
    forM_ prefixes $ \prefix ->
        copyMatchingLibs srcDir dstDir prefix

{- | Create missing .so symlinks for versioned libraries
e.g., libfoo.so.1.2.3 -> libfoo.so.1 -> libfoo.so
-}
createLibrarySymlinks :: FilePath -> Sh ()
createLibrarySymlinks libDir = do
    exists <- test_d libDir
    when exists $ do
        echoErr ":: Creating library symlinks"
        files <- findFiles libDir "*.so.*"
        forM_ files $ \f -> do
            let basename = filename f
                baseTxt = toTextIgnore basename
                -- Extract base name (libfoo from libfoo.so.1.2.3)
                libName = T.takeWhile (/= '.') baseTxt
                soName = libName <> ".so"
                targetPath = libDir </> fromText soName

            -- Create base .so symlink if missing (libfoo.so -> libfoo.so.1.2.3)
            unlessM (test_e targetPath) $
                errExit False $
                    run_ "ln" ["-sf", baseTxt, toTextIgnore targetPath]

            -- Create major version symlinks (e.g., libcupti.so.13 -> libcupti.so.13.0)
            createMajorVersionSymlink libDir basename

-- | Create major version symlink (libfoo.so.9 -> libfoo.so.9.1.2)
createMajorVersionSymlink :: FilePath -> FilePath -> Sh ()
createMajorVersionSymlink libDir lib = do
    let baseTxt = toTextIgnore lib
        parts = T.splitOn "." baseTxt
    -- parts = ["libcupti", "so", "13", "0"] for libcupti.so.13.0
    case parts of
        (name : "so" : major : _ : _) -> do
            let majorName = name <> ".so." <> major
                majorPath = libDir </> fromText majorName
            unlessM (test_e majorPath) $
                when (majorName /= baseTxt) $
                    errExit False $
                        run_ "ln" ["-sf", baseTxt, toTextIgnore majorPath]
        _ -> pure ()

-- | unless for monadic predicates
unlessM :: (Monad m) => m Bool -> m () -> m ()
unlessM cond action = cond >>= \b -> when (not b) action

-- ============================================================================
-- ELF Patching
-- ============================================================================

-- | Patch all ELF binaries in a directory
patchElfBinaries :: FilePath -> Sh ()
patchElfBinaries dir = do
    echoErr ":: Patching ELF binaries"

    -- Make all files writable first (needed for patchelf)
    errExit False $ run_ "chmod" ["-R", "u+w", toTextIgnore dir]

    -- Find all ELF files
    output <- errExit False $ run "find" [toTextIgnore dir, "-type", "f", "(", "-executable", "-o", "-name", "*.so*", ")"]
    let files = filter (not . T.null) $ T.lines output

    forM_ files $ \f -> do
        -- Skip symlinks
        isLink <- errExit False $ run "test" ["-L", f]
        linkCode <- exitCode
        when (linkCode /= 0) $ do
            -- Check if ELF
            fileType <- errExit False $ run "file" ["-b", f]
            when (T.isInfixOf "ELF" fileType) $ do
                -- Calculate relative RPATH
                let fileDir = T.dropWhileEnd (/= '/') f
                    outDir = toTextIgnore dir
                -- Set RPATH
                errExit False $
                    run_
                        "patchelf"
                        [ "--set-rpath"
                        , "$ORIGIN:$ORIGIN/../lib64:$ORIGIN/../lib:$ORIGIN/../nvvm/lib64"
                        , f
                        ]

                -- Set interpreter for executables
                when (T.isInfixOf "executable" fileType) $
                    errExit False $
                        run_
                            "patchelf"
                            [ "--set-interpreter"
                            , "/lib64/ld-linux-x86-64.so.2"
                            , f
                            ]
