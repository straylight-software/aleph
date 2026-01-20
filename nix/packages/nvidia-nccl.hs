{-# LANGUAGE OverloadedStrings #-}

-- | NVIDIA NCCL - Multi-GPU communication library
module Pkg where

import Aleph.Nix.Package
import Prelude hiding (unzip)

pkg :: Drv
pkg =
    mkDerivation
        [ pname "nvidia-nccl"
        , version "2.28.9"
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
        , description "NVIDIA NCCL 2.28.9 - Multi-GPU collective communication"
        , homepage "https://developer.nvidia.com/nccl"
        , license "unfree"
        ]
