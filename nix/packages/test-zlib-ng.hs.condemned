{-# LANGUAGE OverloadedStrings #-}

{- | zlib-ng with typed CMake configuration

Usage: zlib-ng = call-package ./test-zlib-ng.hs {};

This is a test variant that exercises the CMake typed actions.
-}
module Pkg where

import qualified Aleph.Nix.CMake as CMake
import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg =
    defaultDrvSpec
        { pname = "test-zlib-ng"
        , version = "2.2.4"
        , specSrc =
            SrcGitHub
                GitHubSrc
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
        , phases =
            emptyPhases
                { configure =
                    [ CMake.configureAction
                        CMake.defaults
                            { CMake.buildType = Just CMake.Release
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
                { description = "zlib for next generation systems (test variant)"
                , homepage = Just "https://github.com/zlib-ng/zlib-ng"
                , license = "zlib"
                , maintainers = []
                , platforms = []
                }
        }
