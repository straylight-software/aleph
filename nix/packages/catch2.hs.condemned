{-# LANGUAGE OverloadedStrings #-}

-- | Catch2 - Modern C++ test framework
module Pkg where

import qualified Aleph.Nix.CMake as CMake
import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg =
    defaultDrvSpec
        { pname = "catch2"
        , version = "3.3.2"
        , specSrc =
            SrcGitHub
                GitHubSrc
                    { ghOwner = "catchorg"
                    , ghRepo = "Catch2"
                    , ghRev = "v3.3.2"
                    , ghHash = "sha256-t/4iCrzPeDZNNlgibVqx5rhe+d3lXwm1GmBMDDId0VQ="
                    }
        , deps =
            [ buildDep "cmake"
            , buildDep "ninja"
            ]
        , phases =
            emptyPhases
                { configure =
                    [ CMake.configureAction
                        CMake.defaults
                            { CMake.buildType = Just CMake.RelWithDebInfo
                            , CMake.cxxStandard = Just 17
                            , CMake.buildStaticLibs = Just True
                            , CMake.buildSharedLibs = Just False
                            }
                        Ninja
                    ]
                , build = [CMake.buildAction]
                , install = [CMake.installAction]
                }
        , meta =
            Meta
                { description = "Modern C++ test framework"
                , homepage = Just "https://github.com/catchorg/Catch2"
                , license = "bsl-1.0"
                , maintainers = []
                , platforms = []
                }
        }
