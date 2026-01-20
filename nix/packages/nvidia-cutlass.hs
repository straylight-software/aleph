{-# LANGUAGE OverloadedStrings #-}

-- | NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines
module Pkg where

import Aleph.Nix.Package

pkg :: Drv
pkg =
    mkDerivation
        [ pname "nvidia-cutlass"
        , version "4.3.3"
        , src $
            fetchurl
                [ url "https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.3.3.zip"
                , hash "sha256-JGSBZqPafqpbIeF3VfxjiZW9B1snmi0Q13fk+HrpN6w="
                ]
        , nativeBuildInputs ["unzip"]
        , postInstall
            [ mkdir "include"
            , copy "include/cutlass" "include/cutlass"
            , copy "include/cute" "include/cute"
            ]
        , description "NVIDIA CUTLASS 4.3.3 - CUDA linear algebra templates"
        , homepage "https://github.com/NVIDIA/cutlass"
        , license "bsd3"
        ]
