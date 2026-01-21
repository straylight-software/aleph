{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Nvidia.Wheel
Description : Extract NVIDIA libraries from PyPI wheels

PyPI wheels from pypi.nvidia.com contain NVIDIA libraries without
redistribution restrictions. This module provides typed extraction.

== Wheel Structure

Wheels are zip files with a specific layout:

@
nvidia_nccl_cu13-2.28.9-py3-none-manylinux_2_18_x86_64.whl
├── nvidia/
│   └── nccl/
│       ├── lib/
│       │   └── libnccl.so.2
│       └── include/
│           └── nccl.h
└── nvidia_nccl_cu13-2.28.9.dist-info/
    └── METADATA
@

== Usage

@
import Aleph.Script
import qualified Aleph.Script.Nvidia.Wheel as Wheel
import qualified Aleph.Script.Nvidia.Versions as V

main = script $ do
  -- Extract NCCL for current platform
  case Wheel.forPlatform V.currentPlatform Wheel.nccl of
    Just spec -> Wheel.extractToNixLayout spec "/tmp/nccl"
    Nothing -> echoErr "NCCL not available for this platform"

  -- Or extract all available wheels
  Wheel.extractAllWheels V.currentPlatform "/tmp/nvidia-libs"
@
-}
module Aleph.Script.Nvidia.Wheel (
    -- * Wheel specifications
    WheelSpec (..),
    WheelPaths (..),
    WheelDef (..),

    -- * Known wheels (multi-platform)
    nccl,
    cudnn,
    tensorrt,
    cutensor,
    cusparselt,
    nvcomp,
    allWheels,

    -- * Platform selection
    forPlatform,

    -- * Extraction
    download,
    extract,
    extractToNixLayout,
    extractAllWheels,

    -- * Utilities
    wheelUrl,
    wheelFilename,
) where

import Aleph.Script hiding (FilePath)
import Aleph.Script.Nvidia.Versions (Platform (..))
import Control.Monad (forM_, when)
import qualified Data.Text as T

-- ============================================================================
-- Types
-- ============================================================================

-- | Specification for a PyPI wheel (concrete, for one platform)
data WheelSpec = WheelSpec
    { wheelName :: Text
    -- ^ Package name (e.g., "nvidia_nccl_cu13")
    , wheelVersion :: Text
    -- ^ Version (e.g., "2.28.9")
    , wheelPyVersion :: Text
    -- ^ Python version tag (e.g., "py3")
    , wheelAbi :: Text
    -- ^ ABI tag (e.g., "none")
    , wheelPlatform :: Text
    -- ^ Platform tag (e.g., "manylinux_2_18_x86_64")
    , wheelPaths :: WheelPaths
    -- ^ Paths inside the wheel
    , wheelHash :: Text
    -- ^ SHA256 hash (SRI format)
    }
    deriving (Show, Eq)

-- | Paths to lib/include inside the wheel
data WheelPaths = WheelPaths
    { libPath :: Maybe Text
    -- ^ Path to library directory (e.g., "nvidia/nccl/lib")
    , includePath :: Maybe Text
    -- ^ Path to include directory (e.g., "nvidia/nccl/include")
    }
    deriving (Show, Eq)

-- | Multi-platform wheel definition
data WheelDef = WheelDef
    { defName :: Text
    -- ^ Human-readable name (e.g., "nccl")
    , defX86_64 :: Maybe WheelSpec
    -- ^ x86_64-linux wheel spec
    , defAarch64 :: Maybe WheelSpec
    -- ^ aarch64-linux wheel spec
    }
    deriving (Show)

-- ============================================================================
-- Platform Selection
-- ============================================================================

-- | Get wheel spec for a platform, if available
forPlatform :: Platform -> WheelDef -> Maybe WheelSpec
forPlatform X86_64Linux def = defX86_64 def
forPlatform Aarch64Linux def = defAarch64 def
forPlatform (Unsupported _) _ = Nothing

-- ============================================================================
-- Known Wheels (CUDA 13)
-- ============================================================================

-- | NCCL 2.28.9
nccl :: WheelDef
nccl =
    WheelDef
        { defName = "nccl"
        , defX86_64 =
            Just
                WheelSpec
                    { wheelName = "nvidia_nccl_cu13"
                    , wheelVersion = "2.28.9"
                    , wheelPyVersion = "py3"
                    , wheelAbi = "none"
                    , wheelPlatform = "manylinux_2_18_x86_64"
                    , wheelPaths =
                        WheelPaths
                            { libPath = Just "nvidia/nccl/lib"
                            , includePath = Just "nvidia/nccl/include"
                            }
                    , wheelHash = "sha256-5FU6MPNBlfP6HaAqbaPWM30o8gA5Q6oKPSR7vCX+/EI="
                    }
        , defAarch64 = Nothing -- Not available on PyPI
        }

-- | cuDNN 9.17.0.29
cudnn :: WheelDef
cudnn =
    WheelDef
        { defName = "cudnn"
        , defX86_64 =
            Just
                WheelSpec
                    { wheelName = "nvidia_cudnn_cu13"
                    , wheelVersion = "9.17.0.29"
                    , wheelPyVersion = "py3"
                    , wheelAbi = "none"
                    , wheelPlatform = "manylinux_2_27_x86_64"
                    , wheelPaths =
                        WheelPaths
                            { libPath = Just "nvidia/cudnn/lib"
                            , includePath = Just "nvidia/cudnn/include"
                            }
                    , wheelHash = "sha256-Q0Uu8Jj0Q890hyvHj8zZon4af4NWelF5m/rsVgvP4Vo="
                    }
        , defAarch64 = Nothing
        }

-- | TensorRT 10.14.1.48 (libs only)
tensorrt :: WheelDef
tensorrt =
    WheelDef
        { defName = "tensorrt"
        , defX86_64 =
            Just
                WheelSpec
                    { wheelName = "tensorrt_cu13_libs"
                    , wheelVersion = "10.14.1.48"
                    , wheelPyVersion = "py2.py3"
                    , wheelAbi = "none"
                    , wheelPlatform = "manylinux_2_28_x86_64"
                    , wheelPaths =
                        WheelPaths
                            { libPath = Just "tensorrt_libs"
                            , includePath = Nothing
                            }
                    , wheelHash = "sha256-k8SI67WjD/g+pTYD54GAFN5bkyj7JJZZY9I2gUB2UHY="
                    }
        , defAarch64 = Nothing
        }

-- | cuTensor 2.4.1
cutensor :: WheelDef
cutensor =
    WheelDef
        { defName = "cutensor"
        , defX86_64 =
            Just
                WheelSpec
                    { wheelName = "cutensor_cu13"
                    , wheelVersion = "2.4.1"
                    , wheelPyVersion = "py3"
                    , wheelAbi = "none"
                    , wheelPlatform = "manylinux2014_x86_64"
                    , wheelPaths =
                        WheelPaths
                            { libPath = Just "cutensor/lib"
                            , includePath = Just "cutensor/include"
                            }
                    , wheelHash = "sha256-Hz1oTgSVOuRJI7ZzotQVbdmaghQAxC/ocqqF+PFmtyg="
                    }
        , defAarch64 = Nothing
        }

-- | cuSPARSELt 0.8.1
cusparselt :: WheelDef
cusparselt =
    WheelDef
        { defName = "cusparselt"
        , defX86_64 =
            Just
                WheelSpec
                    { wheelName = "nvidia_cusparselt_cu13"
                    , wheelVersion = "0.8.1"
                    , wheelPyVersion = "py3"
                    , wheelAbi = "none"
                    , wheelPlatform = "manylinux2014_x86_64"
                    , wheelPaths =
                        WheelPaths
                            { libPath = Just "nvidia/cusparselt/lib"
                            , includePath = Just "nvidia/cusparselt/include"
                            }
                    , wheelHash = "sha256-eGzodWjDA/rbWvzHEC1FTNMEDXX2+GJvXbRg0YcfTdA="
                    }
        , defAarch64 = Nothing
        }

-- | nvCOMP 5.1.0.21 (Python module, not C library - limited use)
nvcomp :: WheelDef
nvcomp =
    WheelDef
        { defName = "nvcomp"
        , defX86_64 =
            Just
                WheelSpec
                    { wheelName = "nvidia_nvcomp_cu13"
                    , wheelVersion = "5.1.0.21"
                    , wheelPyVersion = "py3"
                    , wheelAbi = "none"
                    , wheelPlatform = "manylinux_2_28_x86_64"
                    , wheelPaths =
                        WheelPaths
                            { libPath = Nothing -- Python module, not C lib
                            , includePath = Nothing
                            }
                    , wheelHash = "sha256-uLifFENVKbdQ8vq2HDVlXiNGEYB+CFfWBsd8QYB+XVg="
                    }
        , defAarch64 = Nothing
        }

-- | All wheel definitions (for bulk extraction)
allWheels :: [WheelDef]
allWheels = [nccl, cudnn, tensorrt, cutensor, cusparselt]

-- Note: nvcomp excluded from allWheels as it's a Python module, not C library

-- ============================================================================
-- URL Construction
-- ============================================================================

-- | Construct the filename for a wheel
wheelFilename :: WheelSpec -> Text
wheelFilename WheelSpec{..} =
    wheelName
        <> "-"
        <> wheelVersion
        <> "-"
        <> wheelPyVersion
        <> "-"
        <> wheelAbi
        <> "-"
        <> wheelPlatform
        <> ".whl"

-- | Construct the PyPI URL for a wheel
wheelUrl :: WheelSpec -> Text
wheelUrl spec@WheelSpec{..} =
    "https://pypi.nvidia.com/"
        <> T.replace "_" "-" wheelName
        <> "/"
        <> wheelFilename spec

-- ============================================================================
-- Download
-- ============================================================================

-- | Download a wheel to the specified path
download :: WheelSpec -> FilePath -> Sh FilePath
download spec outputPath = do
    let url = wheelUrl spec
        filename = wheelFilename spec
        dest = outputPath </> fromText filename

    echoErr $ ":: Downloading " <> filename
    run_ "curl" ["-fsSL", "-o", toTextIgnore dest, url]

    pure dest

-- ============================================================================
-- Extraction
-- ============================================================================

-- | Extract a wheel to a directory
extract :: WheelSpec -> FilePath -> Sh ()
extract spec outputDir = do
    echoErr $ ":: Extracting " <> wheelName spec <> " " <> wheelVersion spec

    withTmpDir $ \tmpDir -> do
        -- Download
        wheelPath <- download spec tmpDir

        -- Extract
        let extractDir = tmpDir </> "extracted"
        mkdirP extractDir
        run_ "unzip" ["-q", toTextIgnore wheelPath, "-d", toTextIgnore extractDir]

        -- Copy lib
        case libPath (wheelPaths spec) of
            Just lp -> do
                let srcLib = extractDir </> fromText lp
                hasLib <- test_d srcLib
                when hasLib $ do
                    mkdirP (outputDir </> "lib")
                    run_ "cp" ["-r", toTextIgnore srcLib <> "/.", toTextIgnore (outputDir </> "lib") <> "/"]
            Nothing -> pure ()

        -- Copy include
        case includePath (wheelPaths spec) of
            Just ip -> do
                let srcInc = extractDir </> fromText ip
                hasInc <- test_d srcInc
                when hasInc $ do
                    mkdirP (outputDir </> "include")
                    run_ "cp" ["-r", toTextIgnore srcInc <> "/.", toTextIgnore (outputDir </> "include") <> "/"]
            Nothing -> pure ()

        -- Create lib64 symlink
        hasLib <- test_d (outputDir </> "lib")
        hasLib64 <- test_e (outputDir </> "lib64")
        when (hasLib && not hasLib64) $ do
            symlink "lib" (outputDir </> "lib64")

    echoErr $ ":: Extracted to " <> toTextIgnore outputDir

-- | Extract a wheel to Nix-style layout with patchelf
extractToNixLayout :: WheelSpec -> FilePath -> Sh ()
extractToNixLayout spec outputDir = do
    extract spec outputDir

    -- Make writable for patchelf
    errExit False $ run_ "chmod" ["-R", "u+w", toTextIgnore outputDir]

    -- Patch ELF binaries
    echoErr ":: Patching ELF binaries..."
    patchElf outputDir
  where
    patchElf :: FilePath -> Sh ()
    patchElf dir = do
        files <- findElfFiles dir
        forM_ files $ \f -> do
            errExit False $ do
                -- Set RPATH relative to $ORIGIN
                run_
                    "patchelf"
                    [ "--set-rpath"
                    , "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib64"
                    , toTextIgnore f
                    ]

    findElfFiles :: FilePath -> Sh [FilePath]
    findElfFiles dir = do
        output <-
            errExit False $
                run
                    "find"
                    [ toTextIgnore dir
                    , "-type"
                    , "f"
                    , "-name"
                    , "*.so*"
                    ]
        pure $ map fromText $ filter (not . T.null) $ T.lines output

-- | Extract all available wheels for a platform
extractAllWheels :: Platform -> FilePath -> Sh ()
extractAllWheels platform outputDir = do
    echoErr $ ":: Extracting all wheels to " <> toTextIgnore outputDir
    mkdirP outputDir

    forM_ allWheels $ \def ->
        case forPlatform platform def of
            Just spec -> do
                let pkgDir = outputDir </> fromText (defName def)
                echoErr $ ":: Processing " <> defName def <> "..."
                mkdirP pkgDir
                extractToNixLayout spec pkgDir
            Nothing ->
                echoErr $ ":: Skipping " <> defName def <> " (not available for platform)"

    echoErr $ ":: Done! All available wheels extracted to " <> toTextIgnore outputDir
