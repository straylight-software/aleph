{-# LANGUAGE OverloadedStrings #-}

-- | zlib-ng - Next generation zlib with SIMD optimizations
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg = defaultDrvSpec
    { pname = "zlib-ng"
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
                (RefSrc Nothing)           -- srcDir
                (RefRel "build")           -- buildDir
                (RefOut "out" Nothing)     -- installPrefix
                "Release"                  -- buildType
                [ "-DBUILD_STATIC_LIBS=ON"
                , "-DBUILD_SHARED_LIBS=OFF"
                , "-DINSTALL_UTILS=ON"
                , "-DZLIB_COMPAT=ON"
                ]
                Ninja
            ]
        , build = [CMakeBuild (RefRel "build") Nothing Nothing]
        , install = [CMakeInstall (RefRel "build")]
        }
    , meta = Meta
        { description = "zlib for next generation systems"
        , homepage = Just "https://github.com/zlib-ng/zlib-ng"
        , license = "zlib"
        , maintainers = []
        , platforms = []
        }
    }
