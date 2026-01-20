{-# LANGUAGE OverloadedStrings #-}

-- | NVIDIA cuSPARSELt - Sparse matrix operations
module Pkg where

import Aleph.Nix.Package
import Prelude hiding (unzip)

pkg :: Drv
pkg =
    mkDerivation
        [ pname "nvidia-cusparselt"
        , version "0.8.1"
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
        , description "NVIDIA cuSPARSELt 0.8.1 - Sparse matrix operations"
        , homepage "https://developer.nvidia.com/cusparselt"
        , license "unfree"
        ]
