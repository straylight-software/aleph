{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Nvidia.Versions
Description : NVIDIA SDK version pins and platform info

Typed version information for NVIDIA libraries. This is the single source
of truth for versions, hashes, and URLs - replacing versions.nix.

== Version Policy

NGC 25.11 is blessed for Blackwell (sm_120). Sources prioritized:

1. PyPI wheels (pypi.nvidia.com) - no redistribution issues
2. Official tarballs - redistribution issues
3. NGC containers - redistribution issues

== Usage

@
import qualified Aleph.Script.Nvidia.Versions as V

main = do
  print $ V.cudaVersion        -- "13.0.1"
  print $ V.wheelUrl V.nccl    -- full URL for NCCL wheel
@
-}
module Aleph.Script.Nvidia.Versions (
    -- * Version strings
    cudaVersion,
    cudnnVersion,
    ncclVersion,
    tensorrtVersion,
    cutensorVersion,
    cusparseltVersion,
    cutlassVersion,
    tritonVersion,

    -- * SM architectures
    SmArch (..),
    volta,
    turing,
    ampere,
    ada,
    hopper,
    blackwell,

    -- * Platform types
    Platform (..),
    currentPlatform,
    x86_64Linux,
    aarch64Linux,

    -- * Container info
    ContainerRef (..),
    tritonserver,
    cudaDevel,

    -- * CUTLASS info
    CutlassInfo (..),
    cutlass,
) where

import Data.Text (Text)
import qualified Data.Text as T
import System.Info (arch, os)

-- ============================================================================
-- Version Strings
-- ============================================================================

-- | CUDA version (from NGC 25.11)
cudaVersion :: Text
cudaVersion = "13.0.1"

-- | cuDNN version
cudnnVersion :: Text
cudnnVersion = "9.17.0.29"

-- | NCCL version
ncclVersion :: Text
ncclVersion = "2.28.9"

-- | TensorRT version
tensorrtVersion :: Text
tensorrtVersion = "10.14.1.48"

-- | cuTensor version
cutensorVersion :: Text
cutensorVersion = "2.4.1"

-- | cuSPARSELt version
cusparseltVersion :: Text
cusparseltVersion = "0.8.1"

-- | CUTLASS version
cutlassVersion :: Text
cutlassVersion = "4.3.3"

-- | Triton Inference Server version
tritonVersion :: Text
tritonVersion = "25.11"

-- ============================================================================
-- SM Architectures
-- ============================================================================

-- | CUDA SM architecture
newtype SmArch = SmArch {smArchName :: Text}
    deriving (Show, Eq)

volta :: SmArch
volta = SmArch "sm_70"

turing :: SmArch
turing = SmArch "sm_75"

ampere :: SmArch
ampere = SmArch "sm_80"

ada :: SmArch
ada = SmArch "sm_89"

hopper :: SmArch
hopper = SmArch "sm_90"

blackwell :: SmArch
blackwell = SmArch "sm_100"

-- ============================================================================
-- Platform
-- ============================================================================

-- | Target platform
data Platform
    = X86_64Linux
    | Aarch64Linux
    | Unsupported Text
    deriving (Show, Eq)

x86_64Linux :: Platform
x86_64Linux = X86_64Linux

aarch64Linux :: Platform
aarch64Linux = Aarch64Linux

-- | Detect current platform from System.Info
currentPlatform :: Platform
currentPlatform = case (arch, os) of
    ("x86_64", "linux") -> X86_64Linux
    ("aarch64", "linux") -> Aarch64Linux
    (a, o) -> Unsupported $ T.pack a <> "-" <> T.pack o

-- ============================================================================
-- Container References
-- ============================================================================

-- | Container image reference with hash
data ContainerRef = ContainerRef
    { containerRef :: Text
    -- ^ Image reference (e.g., "nvcr.io/nvidia/tritonserver:25.11-py3")
    , containerHash :: Text
    -- ^ SHA256 hash (SRI format, empty if unknown)
    , containerVersion :: Text
    -- ^ Version string
    }
    deriving (Show, Eq)

-- | Tritonserver container (has CUDA + cuDNN + NCCL + TensorRT + TensorRT-LLM)
tritonserver :: Platform -> ContainerRef
tritonserver X86_64Linux =
    ContainerRef
        { containerRef = "nvcr.io/nvidia/tritonserver:25.11-py3"
        , containerHash = "sha256-yrTbMURSSc5kx4KTegTErpDjCWcjb9Ehp7pOUtP34pM="
        , containerVersion = tritonVersion
        }
tritonserver Aarch64Linux =
    ContainerRef
        { containerRef = "nvcr.io/nvidia/tritonserver:25.11-py3-igpu"
        , containerHash = "" -- Not yet computed
        , containerVersion = tritonVersion
        }
tritonserver (Unsupported _) =
    ContainerRef
        { containerRef = ""
        , containerHash = ""
        , containerVersion = ""
        }

-- | CUDA devel container (toolkit only, no ML libs)
cudaDevel :: Platform -> ContainerRef
cudaDevel X86_64Linux =
    ContainerRef
        { containerRef = "nvidia/cuda:13.0.1-devel-ubuntu22.04"
        , containerHash = "" -- Not yet computed
        , containerVersion = cudaVersion
        }
cudaDevel Aarch64Linux =
    ContainerRef
        { containerRef = "nvidia/cuda:13.0.1-devel-ubuntu22.04"
        , containerHash = ""
        , containerVersion = cudaVersion
        }
cudaDevel (Unsupported _) =
    ContainerRef
        { containerRef = ""
        , containerHash = ""
        , containerVersion = ""
        }

-- ============================================================================
-- CUTLASS (header-only from GitHub)
-- ============================================================================

-- | CUTLASS download info
data CutlassInfo = CutlassInfo
    { cutlassUrl :: Text
    , cutlassHash :: Text
    , cutlassVer :: Text
    }
    deriving (Show, Eq)

-- | CUTLASS headers from GitHub
cutlass :: CutlassInfo
cutlass =
    CutlassInfo
        { cutlassUrl = "https://github.com/NVIDIA/cutlass/archive/refs/tags/v" <> cutlassVersion <> ".zip"
        , cutlassHash = "sha256-JGSBZqPafqpbIeF3VfxjiZW9B1snmi0Q13fk+HrpN6w="
        , cutlassVer = cutlassVersion
        }
