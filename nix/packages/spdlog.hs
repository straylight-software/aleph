{-# LANGUAGE OverloadedStrings #-}

-- | spdlog - Super fast C++ logging library
module Pkg where

import qualified Aleph.Nix.CMake as CMake
import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg =
    defaultDrvSpec
        { pname = "spdlog"
        , version = "1.15.2"
        , specSrc =
            SrcGitHub
                GitHubSrc
                    { ghOwner = "gabime"
                    , ghRepo = "spdlog"
                    , ghRev = "v1.15.2"
                    , ghHash = "sha256-9RhB4GdFjZbCIfMOWWriLAUf9DE/i/+FTXczr0pD0Vg="
                    }
        , deps =
            [ buildDep "cmake"
            , hostDep "fmt"
            ]
        , phases =
            emptyPhases
                { configure =
                    [ CMake.configureAction
                        CMake.defaults
                            { CMake.buildType = Just CMake.RelWithDebInfo
                            , CMake.cxxStandard = Just 17
                            , CMake.positionIndependentCode = Just True
                            , CMake.buildStaticLibs = Just True
                            , CMake.buildSharedLibs = Just False
                            , CMake.buildExamples = Just False
                            , CMake.buildTesting = Just False
                            , CMake.extraFlags =
                                [ ("SPDLOG_FMT_EXTERNAL_HO", "ON")
                                , ("SPDLOG_BUILD_BENCH", "OFF")
                                ]
                            }
                        Ninja
                    ]
                , build = [CMake.buildAction]
                , install = [CMake.installAction]
                , fixup =
                    [ Mkdir (outSub "share/doc/spdlog") True
                    , Copy (srcSub "example") (outSub "share/doc/spdlog/")
                    ]
                }
        , meta =
            Meta
                { description = "Super fast C++ logging library"
                , homepage = Just "https://github.com/gabime/spdlog"
                , license = "mit"
                , maintainers = []
                , platforms = []
                }
        }
