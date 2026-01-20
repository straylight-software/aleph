{-# LANGUAGE OverloadedStrings #-}

-- | NVIDIA TensorRT - Inference optimization library
module Pkg where

import Aleph.Nix.Package
import Prelude hiding (unzip)

pkg :: Drv
pkg =
    mkDerivation
        [ pname "nvidia-tensorrt"
        , version "10.14.1.48"
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
            , copy "unpacked/tensorrt_libs/." "lib/"
            , symlink "lib" "lib64"
            ]
        , description "NVIDIA TensorRT 10.14.1.48 - High-performance inference"
        , homepage "https://developer.nvidia.com/tensorrt"
        , license "unfree"
        ]
