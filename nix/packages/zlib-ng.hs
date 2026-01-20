{-# LANGUAGE OverloadedStrings #-}

-- | zlib-ng - Next generation zlib with SIMD optimizations
module Pkg where

import Aleph.Nix.Package

pkg :: Drv
pkg =
    mkDerivation
        [ pname "zlib-ng"
        , version "2.2.4"
        , src $
            fetchFromGitHub
                [ owner "zlib-ng"
                , repo "zlib-ng"
                , rev "2.2.4"
                , hash "sha256-Khmrhp5qy4vvoQe4WgoogpjWrgcUB/q8zZeqIydthYg="
                ]
        , nativeBuildInputs ["cmake", "pkg-config"]
        , buildInputs ["gtest"]
        , cmake
            defaults
                { installPrefix = Just "/"
                , buildStaticLibs = Just True
                , buildSharedLibs = Just False
                , extraFlags =
                    [ ("INSTALL_UTILS", "ON")
                    , ("ZLIB_COMPAT", "ON")
                    ]
                }
        , description "zlib for next generation systems"
        , homepage "https://github.com/zlib-ng/zlib-ng"
        , license "zlib"
        ]
