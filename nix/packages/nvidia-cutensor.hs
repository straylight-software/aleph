{-# LANGUAGE OverloadedStrings #-}

-- | NVIDIA cuTensor - Tensor primitive library
module Pkg where

import Aleph.Nix.Package
import Prelude hiding (unzip)

pkg :: Drv
pkg =
    mkDerivation
        [ pname "nvidia-cutensor"
        , version "2.4.1"
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
        , description "NVIDIA cuTensor 2.4.1 - Tensor linear algebra"
        , homepage "https://developer.nvidia.com/cutensor"
        , license "unfree"
        ]
