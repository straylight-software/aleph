{-# LANGUAGE OverloadedStrings #-}

{- | zlib-ng with typed CMake configuration

Usage: zlib-ng = call-package ./test-zlib-ng.hs {};
-}
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
        , cmake
            defaults
                { buildStaticLibs = Just True
                , buildSharedLibs = Just False
                , buildType = Just Release
                }
        , nativeBuildInputs ["cmake", "pkg-config"]
        , buildInputs ["gtest"]
        , description "zlib for next generation systems"
        , homepage "https://github.com/zlib-ng/zlib-ng"
        , license "zlib"
        ]
