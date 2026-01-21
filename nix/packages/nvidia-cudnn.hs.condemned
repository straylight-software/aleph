{-# LANGUAGE OverloadedStrings #-}

-- | NVIDIA cuDNN - Deep learning primitives library
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg =
    defaultDrvSpec
        { pname = "nvidia-cudnn"
        , version = "9.17.0.29"
        , specSrc =
            SrcUrl
                UrlSrc
                    { urlUrl = "https://pypi.nvidia.com/nvidia-cudnn-cu13/nvidia_cudnn_cu13-9.17.0.29-py3-none-manylinux_2_27_x86_64.whl"
                    , urlHash = "sha256-Q0Uu8Jj0Q890hyvHj8zZon4af4NWelF5m/rsVgvP4Vo="
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
                    , Copy (RefRel "unpacked/nvidia/cudnn/lib/.") (outSub "lib/")
                    , Copy (RefRel "unpacked/nvidia/cudnn/include/.") (outSub "include/")
                    , Symlink (RefLit "lib") (outSub "lib64")
                    ]
                }
        , meta =
            Meta
                { description = "NVIDIA cuDNN 9.17.0.29 - Deep neural network library"
                , homepage = Just "https://developer.nvidia.com/cudnn"
                , license = "unfree"
                , maintainers = []
                , platforms = []
                }
        }
