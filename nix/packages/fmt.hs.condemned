{-# LANGUAGE OverloadedStrings #-}

-- | fmt - Small, safe, and fast formatting library for C++
module Pkg where

import qualified Aleph.Nix.CMake as CMake
import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg =
    defaultDrvSpec
        { pname = "fmt"
        , version = "11.2.0"
        , specSrc =
            SrcGitHub
                GitHubSrc
                    { ghOwner = "fmtlib"
                    , ghRepo = "fmt"
                    , ghRev = "11.2.0"
                    , ghHash = "sha256-sAlU5L/olxQUYcv8euVYWTTB8TrVeQgXLHtXy8IMEnU="
                    }
        , deps = [buildDep "cmake"]
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
                            }
                        Ninja
                    ]
                , build = [CMake.buildAction]
                , install = [CMake.installAction]
                }
        , meta =
            Meta
                { description = "Small, safe and fast formatting library"
                , homepage = Just "https://fmt.dev/"
                , license = "mit"
                , maintainers = []
                , platforms = []
                }
        }
