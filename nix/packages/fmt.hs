{-# LANGUAGE OverloadedStrings #-}

-- | fmt - Small, safe, and fast formatting library for C++
module Pkg where

import Aleph.Nix.Package

pkg :: Drv
pkg =
    mkDerivation
        [ pname "fmt"
        , version "11.2.0"
        , src $
            fetchFromGitHub
                [ owner "fmtlib"
                , repo "fmt"
                , rev "11.2.0"
                , hash "sha256-sAlU5L/olxQUYcv8euVYWTTB8TrVeQgXLHtXy8IMEnU="
                ]
        , nativeBuildInputs ["cmake"]
        , cmake
            defaults
                { buildType = Just RelWithDebInfo
                , cxxStandard = Just 17
                , positionIndependentCode = Just True
                , buildStaticLibs = Just True
                , buildSharedLibs = Just False
                }
        , description "Small, safe and fast formatting library"
        , homepage "https://fmt.dev/"
        , license "mit"
        ]
