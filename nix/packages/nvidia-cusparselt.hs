{-# LANGUAGE OverloadedStrings #-}

-- | NVIDIA cuSPARSELt - Sparse matrix operations
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg = defaultDrvSpec
    { pname = "nvidia-cusparselt"
    , version = "0.8.1"
    , specSrc = SrcUrl UrlSrc
        { urlUrl = "https://pypi.nvidia.com/nvidia-cusparselt-cu13/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl"
        , urlHash = "sha256-eGzodWjDA/rbWvzHEC1FTNMEDXX2+GJvXbRg0YcfTdA="
        }
    , deps = 
        [ buildDep "autoPatchelfHook"
        , buildDep "unzip"
        , hostDep "stdenv.cc.cc.lib"
        , hostDep "zlib"
        ]
    , phases = emptyPhases
        { unpack = []  -- Don't auto-unpack wheel
        , install = 
            [ Unzip src (RefRel "unpacked")
            , Mkdir (outSub "lib") True
            , Mkdir (outSub "include") True
            , Copy (RefRel "unpacked/nvidia/cusparselt/lib/.") (outSub "lib/")
            , Copy (RefRel "unpacked/nvidia/cusparselt/include/.") (outSub "include/")
            , Symlink (RefLit "lib") (outSub "lib64")
            ]
        }
    , meta = Meta
        { description = "NVIDIA cuSPARSELt 0.8.1 - Sparse matrix operations"
        , homepage = Just "https://developer.nvidia.com/cusparselt"
        , license = "unfree"
        , maintainers = []
        , platforms = []
        }
    }
