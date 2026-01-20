{-# LANGUAGE OverloadedStrings #-}

-- | Catch2 - Modern C++ test framework
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg = defaultDrvSpec
    { pname = "catch2"
    , version = "3.3.2"
    , specSrc = SrcGitHub GitHubSrc
        { ghOwner = "catchorg"
        , ghRepo = "Catch2"
        , ghRev = "v3.3.2"
        , ghHash = "sha256-t/4iCrzPeDZNNlgibVqx5rhe+d3lXwm1GmBMDDId0VQ="
        }
    , deps = 
        [ buildDep "cmake"
        , buildDep "ninja"
        ]
    , phases = emptyPhases
        { configure = 
            [ CMakeConfigure 
                (RefSrc Nothing)
                (RefRel "build")
                (RefOut "out" Nothing)
                "RelWithDebInfo"
                [ "-DCMAKE_CXX_STANDARD=17"
                , "-DBUILD_STATIC_LIBS=ON"
                , "-DBUILD_SHARED_LIBS=OFF"
                ]
                Ninja
            ]
        , build = [CMakeBuild (RefRel "build") Nothing Nothing]
        , install = [CMakeInstall (RefRel "build")]
        }
    , meta = Meta
        { description = "Modern C++ test framework"
        , homepage = Just "https://github.com/catchorg/Catch2"
        , license = "bsl-1.0"
        , maintainers = []
        , platforms = []
        }
    }
