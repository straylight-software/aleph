{-# LANGUAGE OverloadedStrings #-}

-- | NVIDIA cuDNN - Deep learning primitives library
module Pkg where

import Aleph.Nix.Package
import Prelude hiding (unzip)

pkg :: Drv
pkg =
    mkDerivation
        [ pname "nvidia-cudnn"
        , version "9.17.0.29"
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
        , description "NVIDIA cuDNN 9.17.0.29 - Deep neural network library"
        , homepage "https://developer.nvidia.com/cudnn"
        , license "unfree"
        ]
