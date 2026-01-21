{-# LANGUAGE OverloadedStrings #-}

-- | NVIDIA TensorRT - Inference optimization library
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg =
    defaultDrvSpec
        { pname = "nvidia-tensorrt"
        , version = "10.14.1.48"
        , specSrc =
            SrcUrl
                UrlSrc
                    { urlUrl = "https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-10.14.1.48-py2.py3-none-manylinux_2_28_x86_64.whl"
                    , urlHash = "sha256-k8SI67WjD/g+pTYD54GAFN5bkyj7JJZZY9I2gUB2UHY="
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
                    , Copy (RefRel "unpacked/tensorrt_libs/.") (outSub "lib/")
                    , Symlink (RefLit "lib") (outSub "lib64")
                    ]
                }
        , meta =
            Meta
                { description = "NVIDIA TensorRT 10.14.1.48 - High-performance inference"
                , homepage = Just "https://developer.nvidia.com/tensorrt"
                , license = "unfree"
                , maintainers = []
                , platforms = []
                }
        }
