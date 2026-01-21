{-# LANGUAGE OverloadedStrings #-}

-- | NVIDIA NCCL - Multi-GPU communication library
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg =
    defaultDrvSpec
        { pname = "nvidia-nccl"
        , version = "2.28.9"
        , specSrc =
            SrcUrl
                UrlSrc
                    { urlUrl = "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.28.9-py3-none-manylinux_2_18_x86_64.whl"
                    , urlHash = "sha256-5FU6MPNBlfP6HaAqbaPWM30o8gA5Q6oKPSR7vCX+/EI="
                    }
        , deps =
            [ buildDep "autoPatchelfHook"
            , buildDep "unzip"
            , hostDep "stdenv.cc.cc.lib"
            , hostDep "zlib"
            ]
        , phases =
            emptyPhases
                { unpack = [] -- Don't auto-unpack wheel
                , install =
                    [ Unzip src (RefRel "unpacked")
                    , Mkdir (outSub "lib") True
                    , Mkdir (outSub "include") True
                    , Copy (RefRel "unpacked/nvidia/nccl/lib/.") (outSub "lib/")
                    , Copy (RefRel "unpacked/nvidia/nccl/include/.") (outSub "include/")
                    , Symlink (RefLit "lib") (outSub "lib64")
                    ]
                }
        , meta =
            Meta
                { description = "NVIDIA NCCL 2.28.9 - Multi-GPU collective communication"
                , homepage = Just "https://developer.nvidia.com/nccl"
                , license = "unfree"
                , maintainers = []
                , platforms = []
                }
        }
