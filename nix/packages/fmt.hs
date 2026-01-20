{-# LANGUAGE OverloadedStrings #-}

-- | fmt - Small, safe, and fast formatting library for C++
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg = defaultDrvSpec
    { pname = "fmt"
    , version = "11.2.0"
    , specSrc = SrcGitHub GitHubSrc
        { ghOwner = "fmtlib"
        , ghRepo = "fmt"
        , ghRev = "11.2.0"
        , ghHash = "sha256-sAlU5L/olxQUYcv8euVYWTTB8TrVeQgXLHtXy8IMEnU="
        }
    , deps = [buildDep "cmake"]
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
                ]
                Ninja
            ]
        , build = [CMakeBuild (RefRel "build") Nothing Nothing]
        , install = [CMakeInstall (RefRel "build")]
        }
    , meta = Meta
        { description = "Small, safe and fast formatting library"
        , homepage = Just "https://fmt.dev/"
        , license = "mit"
        , maintainers = []
        , platforms = []
        }
    }
