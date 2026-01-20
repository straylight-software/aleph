{-# LANGUAGE OverloadedStrings #-}

-- | spdlog - Super fast C++ logging library
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg = defaultDrvSpec
    { pname = "spdlog"
    , version = "1.15.2"
    , specSrc = SrcGitHub GitHubSrc
        { ghOwner = "gabime"
        , ghRepo = "spdlog"
        , ghRev = "v1.15.2"
        , ghHash = "sha256-9RhB4GdFjZbCIfMOWWriLAUf9DE/i/+FTXczr0pD0Vg="
        }
    , deps = 
        [ buildDep "cmake"
        , hostDep "fmt"
        ]
    , phases = emptyPhases
        { configure = 
            [ CMakeConfigure 
                (RefSrc Nothing)
                (RefRel "build")
                (RefOut "out" Nothing)
                "RelWithDebInfo"
                [ "-DCMAKE_CXX_STANDARD=17"
                , "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
                , "-DBUILD_STATIC_LIBS=ON"
                , "-DBUILD_SHARED_LIBS=OFF"
                , "-DSPDLOG_FMT_EXTERNAL_HO=ON"
                , "-DSPDLOG_BUILD_EXAMPLE=OFF"
                , "-DSPDLOG_BUILD_BENCH=OFF"
                , "-DSPDLOG_BUILD_TESTS=OFF"
                ]
                Ninja
            ]
        , build = [CMakeBuild (RefRel "build") Nothing Nothing]
        , install = [CMakeInstall (RefRel "build")]
        , fixup = 
            [ Mkdir (RefOut "out" (Just "share/doc/spdlog")) True
            , Copy (RefSrc (Just "example")) (RefOut "out" (Just "share/doc/spdlog/"))
            ]
        }
    , meta = Meta
        { description = "Super fast C++ logging library"
        , homepage = Just "https://github.com/gabime/spdlog"
        , license = "mit"
        , maintainers = []
        , platforms = []
        }
    }
