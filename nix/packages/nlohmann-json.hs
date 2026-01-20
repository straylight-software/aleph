{-# LANGUAGE OverloadedStrings #-}

-- | nlohmann/json - JSON for Modern C++
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg = defaultDrvSpec
    { pname = "nlohmann_json"
    , version = "3.12.0"
    , specSrc = SrcGitHub GitHubSrc
        { ghOwner = "nlohmann"
        , ghRepo = "json"
        , ghRev = "v3.12.0"
        , ghHash = "sha256-cECvDOLxgX7Q9R3IE86Hj9JJUxraDQvhoyPDF03B2CY="
        }
    , deps = 
        [ buildDep "cmake"
        , buildDep "pkg-config"
        ]
    , phases = emptyPhases
        { configure = 
            [ CMakeConfigure 
                (RefSrc Nothing)
                (RefRel "build")
                (RefOut "out" Nothing)
                "Release"
                ["-DJSON_BuildTests=OFF"]
                Ninja
            ]
        , build = [CMakeBuild (RefRel "build") Nothing Nothing]
        , install = [CMakeInstall (RefRel "build")]
        , fixup = 
            [ Remove (RefOut "out" (Just "share/pkgconfig/nlohmann_json.pc")) True
            , Mkdir (RefOut "out" (Just "lib/pkgconfig")) True
            , Write (RefOut "out" (Just "lib/pkgconfig/nlohmann_json.pc")) pkgConfig
            , Symlink (RefLit "nlohmann/json.hpp") (RefOut "out" (Just "include/json.hpp"))
            ]
        }
    , meta = Meta
        { description = "JSON for Modern C++"
        , homepage = Just "https://github.com/nlohmann/json"
        , license = "mit"
        , maintainers = []
        , platforms = []
        }
    }
  where
    pkgConfig =
        "prefix=${out}\n\
        \includedir=${prefix}/include\n\
        \\n\
        \Name: nlohmann_json\n\
        \Description: JSON for Modern C++\n\
        \Version: 3.12.0\n\
        \Cflags: -I${includedir}\n"
