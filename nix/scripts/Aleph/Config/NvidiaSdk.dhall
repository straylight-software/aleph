-- |
-- Module      : Aleph.Config.NvidiaSdk
-- Description : Configuration for NVIDIA SDK extraction
--
-- Typed configuration for nvidia-sdk-extract tool.
-- Supports multiple extraction sources:
--   - Container: Extract from NGC container image
--   - Tarball: Extract from downloadable archive
--   - Installer: Run .run installer in FHS environment
--
-- NGC 25.11 blessed for Blackwell (sm_120)

let Base = ./Base.dhall

-- ============================================================================
-- Version specification
-- ============================================================================

let Versions =
    { Type =
        { cuda : Text
        , cudnn : Text
        , nccl : Text
        , tensorrt : Text
        , cutensor : Text
        }
    , default =
        { cuda = "13.0.2"
        , cudnn = "9.17.0"
        , nccl = "2.28.9"
        , tensorrt = "10.14.1"
        , cutensor = "2.4.1"
        }
    }

-- ============================================================================
-- Source types
-- ============================================================================

-- | Container image source (NGC)
let ContainerSource =
    { imageRef : Base.ImageRef
    -- ^ Container image reference (e.g., "nvcr.io/nvidia/tritonserver:25.11-py3")
    , platform : Text
    -- ^ Target platform (linux/amd64 or linux/arm64)
    }

-- | Tarball source (direct download)
let TarballSource =
    { url : Text
    -- ^ Download URL
    , hash : Text
    -- ^ Expected SHA256 hash (SRI format)
    , stripComponents : Natural
    -- ^ Number of leading path components to strip (like tar --strip-components)
    }

-- | Installer source (.run file)
let InstallerSource =
    { url : Text
    -- ^ Download URL for .run installer
    , hash : Text
    -- ^ Expected SHA256 hash
    }

-- ============================================================================
-- Component configurations
-- ============================================================================

-- | CUDA toolkit configuration
let CudaConfig =
    { Type =
        { version : Text
        , source : InstallerSource
        }
    , default = {=}
    }

-- | cuDNN configuration
let CudnnConfig =
    { Type =
        { version : Text
        , source : TarballSource
        }
    , default = {=}
    }

-- | NCCL configuration
let NcclConfig =
    { Type =
        { version : Text
        , fromContainer : Bool
        -- ^ If true, extract from triton container; if false, use tarball
        , source : Optional TarballSource
        }
    , default =
        { fromContainer = True
        , source = None TarballSource
        }
    }

-- | TensorRT configuration
let TensorrtConfig =
    { Type =
        { version : Text
        , source : TarballSource
        }
    , default = {=}
    }

-- | cuTensor configuration
let CutensorConfig =
    { Type =
        { version : Text
        , source : TarballSource
        }
    , default = {=}
    }

-- | CUTLASS configuration (header-only)
let CutlassConfig =
    { Type =
        { version : Text
        , source : TarballSource
        }
    , default = {=}
    }

-- | Tritonserver configuration
let TritonserverConfig =
    { Type =
        { version : Text
        , source : ContainerSource
        }
    , default = {=}
    }

-- ============================================================================
-- Main SDK configuration
-- ============================================================================

let SdkConfig =
    { Type =
        { outputDir : Text
        -- ^ Where to write extracted SDK
        , versions : Versions.Type
        -- ^ Expected versions for validation
        , tritonContainer : ContainerSource
        -- ^ Container to extract NCCL and base libs from
        , cuda : Optional CudaConfig.Type
        , cudnn : Optional CudnnConfig.Type
        , nccl : Optional NcclConfig.Type
        , tensorrt : Optional TensorrtConfig.Type
        , cutensor : Optional CutensorConfig.Type
        , cutlass : Optional CutlassConfig.Type
        , tritonserver : Optional TritonserverConfig.Type
        }
    , default =
        { versions = Versions.default
        , cuda = None CudaConfig.Type
        , cudnn = None CudnnConfig.Type
        , nccl = None NcclConfig.Type
        , tensorrt = None TensorrtConfig.Type
        , cutensor = None CutensorConfig.Type
        , cutlass = None CutlassConfig.Type
        , tritonserver = None TritonserverConfig.Type
        }
    }

-- ============================================================================
-- Presets for common configurations
-- ============================================================================

-- | NGC 25.11 Blackwell-blessed versions
let ngc-25-11 =
    { versions = Versions::{
        , cuda = "13.0.2"
        , cudnn = "9.17.0"
        , nccl = "2.28.9"
        , tensorrt = "10.14.1"
        , cutensor = "2.4.1"
        }
    , tritonContainer =
        { imageRef = "nvcr.io/nvidia/tritonserver:25.11-py3"
        , platform = "linux/amd64"
        }
    }

-- ============================================================================
-- Export
-- ============================================================================

in  { Versions
    , ContainerSource
    , TarballSource
    , InstallerSource
    , CudaConfig
    , CudnnConfig
    , NcclConfig
    , TensorrtConfig
    , CutensorConfig
    , CutlassConfig
    , TritonserverConfig
    , SdkConfig
    , ngc-25-11
    }
