{-# LANGUAGE OverloadedStrings #-}

{- | zlib-ng with typed CMake configuration

Usage: zlib-ng = call-package ./test-zlib-ng.hs {};

This is a test variant that exercises the CMake typed actions.
-}
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg = defaultDrvSpec
    { pname = "test-zlib-ng"
    , version = "2.2.4"
    , specSrc = SrcGitHub GitHubSrc
        { ghOwner = "zlib-ng"
        , ghRepo = "zlib-ng"
        , ghRev = "2.2.4"
        , ghHash = "sha256-Khmrhp5qy4vvoQe4WgoogpjWrgcUB/q8zZeqIydthYg="
        }
    , deps = 
        [ buildDep "cmake"
        , buildDep "pkg-config"
        , hostDep "gtest"
        ]
    , phases = emptyPhases
        { configure = 
            [ CMakeConfigure 
                src                        -- srcDir
                (RefRel "build")           -- buildDir
                out                        -- installPrefix
                "Release"                  -- buildType
                [ "-DBUILD_STATIC_LIBS=ON"
                , "-DBUILD_SHARED_LIBS=OFF"
                ]
                Ninja
            ]
        , build = [CMakeBuild (RefRel "build") Nothing Nothing]
        , install = [CMakeInstall (RefRel "build")]
        }
    , meta = Meta
        { description = "zlib for next generation systems (test variant)"
        , homepage = Just "https://github.com/zlib-ng/zlib-ng"
        , license = "zlib"
        , maintainers = []
        , platforms = []
        }
    }
