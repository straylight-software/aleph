{-# LANGUAGE OverloadedStrings #-}

-- | Catch2 - Modern C++ test framework
module Pkg where

import Aleph.Nix.Package

pkg :: Drv
pkg =
    mkDerivation
        [ pname "catch2"
        , version "3.3.2"
        , src $
            fetchFromGitHub
                [ owner "catchorg"
                , repo "Catch2"
                , rev "v3.3.2"
                , hash "sha256-t/4iCrzPeDZNNlgibVqx5rhe+d3lXwm1GmBMDDId0VQ="
                ]
        , nativeBuildInputs ["cmake", "ninja"]
        , cmake
            defaults
                { buildType = Just RelWithDebInfo
                , cxxStandard = Just 17
                , buildStaticLibs = Just True
                , buildSharedLibs = Just False
                }
        , description "Modern C++ test framework"
        , homepage "https://github.com/catchorg/Catch2"
        , license "bsl-1.0"
        ]
