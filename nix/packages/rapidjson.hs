{-# LANGUAGE OverloadedStrings #-}

-- | RapidJSON - Fast JSON parser/generator for C++
module Pkg where

import Aleph.Nix.DrvSpec

pkg :: DrvSpec
pkg = defaultDrvSpec
    { pname = "rapidjson"
    , version = "unstable-2024-04-09"
    , specSrc = SrcGitHub GitHubSrc
        { ghOwner = "Tencent"
        , ghRepo = "rapidjson"
        , ghRev = "ab1842a2dae061284c0a62dca1cc6d5e7e37e346"
        , ghHash = "sha256-kAGVJfDHEUV2qNR1LpnWq3XKBJy4hD3Swh6LX5shJpM="
        }
    , deps = 
        [ buildDep "cmake"
        , buildDep "doxygen"
        , buildDep "graphviz"
        , hostDep "gtest"
        , checkDep "valgrind"
        ]
    , phases = emptyPhases
        { patch =
            [ Substitute (RefSrc (Just "doc/Doxyfile.in"))
                [("WARN_IF_UNDOCUMENTED   = YES", "WARN_IF_UNDOCUMENTED   = NO")]
            , Substitute (RefSrc (Just "doc/Doxyfile.zh-cn.in"))
                [("WARN_IF_UNDOCUMENTED   = YES", "WARN_IF_UNDOCUMENTED   = NO")]
            ]
        , configure = 
            [ CMakeConfigure 
                (RefSrc Nothing)
                (RefRel "build")
                (RefOut "out" Nothing)
                "Release"
                [ "-DRAPIDJSON_BUILD_DOC=ON"
                , "-DRAPIDJSON_BUILD_TESTS=ON"
                , "-DRAPIDJSON_BUILD_EXAMPLES=ON"
                , "-DRAPIDJSON_BUILD_CXX11=OFF"
                , "-DRAPIDJSON_BUILD_CXX17=ON"
                , "-DRAPIDJSON_ENABLE_INSTRUMENTATION_OPT=OFF"
                , "-DCMAKE_CXX_FLAGS_RELEASE=-Wno-error"
                ]
                Ninja
            ]
        , build = [CMakeBuild (RefRel "build") Nothing Nothing]
        , install = [CMakeInstall (RefRel "build")]
        }
    , meta = Meta
        { description = "Fast JSON parser/generator for C++ with SAX/DOM style API"
        , homepage = Just "http://rapidjson.org/"
        , license = "mit"
        , maintainers = []
        , platforms = []
        }
    }
