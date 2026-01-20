{-# LANGUAGE OverloadedStrings #-}

-- | NVIDIA cuTensor - Tensor primitive library
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg = defaultDrvSpec
    { pname = "nvidia-cutensor"
    , version = "2.4.1"
    , specSrc = SrcUrl UrlSrc
        { urlUrl = "https://pypi.nvidia.com/cutensor-cu13/cutensor_cu13-2.4.1-py3-none-manylinux2014_x86_64.whl"
        , urlHash = "sha256-Hz1oTgSVOuRJI7ZzotQVbdmaghQAxC/ocqqF+PFmtyg="
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
            , Copy (RefRel "unpacked/cutensor/lib/.") (outSub "lib/")
            , Copy (RefRel "unpacked/cutensor/include/.") (outSub "include/")
            , Symlink (RefLit "lib") (outSub "lib64")
            ]
        }
    , meta = Meta
        { description = "NVIDIA cuTensor 2.4.1 - Tensor linear algebra"
        , homepage = Just "https://developer.nvidia.com/cutensor"
        , license = "unfree"
        , maintainers = []
        , platforms = []
        }
    }
