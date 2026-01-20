{-# LANGUAGE OverloadedStrings #-}

{- | NVIDIA SDK package definitions.

Packages extracted from PyPI wheels (preferred - no redistribution issues).

= Wheel Layout

PyPI wheels from pypi.nvidia.com contain libraries in a predictable structure:

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

= autoPatchelfHook

All packages use autoPatchelfHook for proper library resolution.
The buildInputs list provides runtime dependencies; autoPatchelfIgnoreMissingDeps
handles driver libs (libcuda.so.1) provided at runtime.
-}
module Aleph.Nix.Packages.Nvidia (
    -- * Versions
    ncclVersion,
    cudnnVersion,
    tensorrtVersion,
    cutensorVersion,
    cusparseltVersion,
    cutlassVersion,

    -- * Wheel packages
    nccl,
    cudnn,
    tensorrt,
    cutensor,
    cusparselt,

    -- * Header-only
    cutlass,
) where

import Aleph.Nix.Derivation (Drv)
import Aleph.Nix.Syntax
import Data.Text (Text)
import Prelude hiding (unzip)

-- ============================================================================
-- Versions (NGC 25.11 blessed for Blackwell sm_120)
-- ============================================================================

ncclVersion, cudnnVersion, tensorrtVersion, cutensorVersion, cusparseltVersion, cutlassVersion :: Text
ncclVersion = "2.28.9"
cudnnVersion = "9.17.0.29"
tensorrtVersion = "10.14.1.48"
cutensorVersion = "2.4.1"
cusparseltVersion = "0.8.1"
cutlassVersion = "4.3.3"

-- ============================================================================
-- Wheel packages
-- ============================================================================

-- | NVIDIA NCCL - Multi-GPU communication
nccl :: Drv
nccl =
    mkDerivation
        [ pname "nvidia-nccl"
        , version ncclVersion
        , src $
            fetchurl
                [ url "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.28.9-py3-none-manylinux_2_18_x86_64.whl"
                , hash "sha256-5FU6MPNBlfP6HaAqbaPWM30o8gA5Q6oKPSR7vCX+/EI="
                ]
        , dontUnpack True
        , nativeBuildInputs ["autoPatchelfHook", "unzip"]
        , buildInputs ["stdenv.cc.cc.lib", "zlib"]
        , installPhase
            [ unzip "unpacked"
            , mkdir "lib"
            , mkdir "include"
            , copy "unpacked/nvidia/nccl/lib/." "lib/"
            , copy "unpacked/nvidia/nccl/include/." "include/"
            , symlink "lib" "lib64"
            ]
        , description $ "NVIDIA NCCL " <> ncclVersion <> " (from PyPI)"
        , homepage "https://developer.nvidia.com/nccl"
        , license "unfree"
        ]

-- | NVIDIA cuDNN - Deep learning primitives
cudnn :: Drv
cudnn =
    mkDerivation
        [ pname "nvidia-cudnn"
        , version cudnnVersion
        , src $
            fetchurl
                [ url "https://pypi.nvidia.com/nvidia-cudnn-cu13/nvidia_cudnn_cu13-9.17.0.29-py3-none-manylinux_2_27_x86_64.whl"
                , hash "sha256-Q0Uu8Jj0Q890hyvHj8zZon4af4NWelF5m/rsVgvP4Vo="
                ]
        , dontUnpack True
        , nativeBuildInputs ["autoPatchelfHook", "unzip"]
        , buildInputs ["stdenv.cc.cc.lib", "zlib"]
        , installPhase
            [ unzip "unpacked"
            , mkdir "lib"
            , mkdir "include"
            , copy "unpacked/nvidia/cudnn/lib/." "lib/"
            , copy "unpacked/nvidia/cudnn/include/." "include/"
            , symlink "lib" "lib64"
            ]
        , description $ "NVIDIA cuDNN " <> cudnnVersion <> " (from PyPI)"
        , homepage "https://developer.nvidia.com/cudnn"
        , license "unfree"
        ]

-- | NVIDIA TensorRT - Inference optimization
tensorrt :: Drv
tensorrt =
    mkDerivation
        [ pname "nvidia-tensorrt"
        , version tensorrtVersion
        , src $
            fetchurl
                [ url "https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-10.14.1.48-py2.py3-none-manylinux_2_28_x86_64.whl"
                , hash "sha256-k8SI67WjD/g+pTYD54GAFN5bkyj7JJZZY9I2gUB2UHY="
                ]
        , dontUnpack True
        , nativeBuildInputs ["autoPatchelfHook", "unzip"]
        , buildInputs ["stdenv.cc.cc.lib", "zlib"]
        , installPhase
            [ unzip "unpacked"
            , mkdir "lib"
            , copy "unpacked/tensorrt_libs/." "lib/" -- libs-only wheel
            , symlink "lib" "lib64"
            ]
        , description $ "NVIDIA TensorRT " <> tensorrtVersion <> " (from PyPI)"
        , homepage "https://developer.nvidia.com/tensorrt"
        , license "unfree"
        ]

-- | NVIDIA cuTensor - Tensor operations
cutensor :: Drv
cutensor =
    mkDerivation
        [ pname "nvidia-cutensor"
        , version cutensorVersion
        , src $
            fetchurl
                [ url "https://pypi.nvidia.com/cutensor-cu13/cutensor_cu13-2.4.1-py3-none-manylinux2014_x86_64.whl"
                , hash "sha256-Hz1oTgSVOuRJI7ZzotQVbdmaghQAxC/ocqqF+PFmtyg="
                ]
        , dontUnpack True
        , nativeBuildInputs ["autoPatchelfHook", "unzip"]
        , buildInputs ["stdenv.cc.cc.lib", "zlib"]
        , installPhase
            [ unzip "unpacked"
            , mkdir "lib"
            , mkdir "include"
            , copy "unpacked/cutensor/lib/." "lib/"
            , copy "unpacked/cutensor/include/." "include/"
            , symlink "lib" "lib64"
            ]
        , description $ "NVIDIA cuTensor " <> cutensorVersion <> " (from PyPI)"
        , homepage "https://developer.nvidia.com/cutensor"
        , license "unfree"
        ]

-- | NVIDIA cuSPARSELt - Sparse matrix operations
cusparselt :: Drv
cusparselt =
    mkDerivation
        [ pname "nvidia-cusparselt"
        , version cusparseltVersion
        , src $
            fetchurl
                [ url "https://pypi.nvidia.com/nvidia-cusparselt-cu13/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl"
                , hash "sha256-eGzodWjDA/rbWvzHEC1FTNMEDXX2+GJvXbRg0YcfTdA="
                ]
        , dontUnpack True
        , nativeBuildInputs ["autoPatchelfHook", "unzip"]
        , buildInputs ["stdenv.cc.cc.lib", "zlib"]
        , installPhase
            [ unzip "unpacked"
            , mkdir "lib"
            , mkdir "include"
            , copy "unpacked/nvidia/cusparselt/lib/." "lib/"
            , copy "unpacked/nvidia/cusparselt/include/." "include/"
            , symlink "lib" "lib64"
            ]
        , description $ "NVIDIA cuSPARSELt " <> cusparseltVersion <> " (from PyPI)"
        , homepage "https://developer.nvidia.com/cusparselt"
        , license "unfree"
        ]

-- ============================================================================
-- Header-only
-- ============================================================================

-- | NVIDIA CUTLASS - CUDA Templates for Linear Algebra
cutlass :: Drv
cutlass =
    mkDerivation
        [ pname "nvidia-cutlass"
        , version cutlassVersion
        , src $
            fetchurl
                [ url "https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.3.3.zip"
                , hash "sha256-JGSBZqPafqpbIeF3VfxjiZW9B1snmi0Q13fk+HrpN6w="
                ]
        , nativeBuildInputs ["unzip"]
        , postInstall
            [ mkdir "include"
            , copy "include/cutlass" "include/cutlass"
            , copy "include/cute" "include/cute"
            ]
        , description $ "NVIDIA CUTLASS " <> cutlassVersion <> " (header-only)"
        , homepage "https://github.com/NVIDIA/cutlass"
        , license "bsd3"
        ]
