{-# LANGUAGE OverloadedStrings #-}

-- | RapidJSON - Fast JSON parser/generator for C++
module Pkg where

import Aleph.Nix.Package
import qualified Aleph.Nix.Tools.Substitute as Sub

pkg :: Drv
pkg =
    mkDerivation
        [ pname "rapidjson"
        , version "unstable-2024-04-09"
        , src $
            fetchFromGitHub
                [ owner "Tencent"
                , repo "rapidjson"
                , rev "ab1842a2dae061284c0a62dca1cc6d5e7e37e346"
                , hash "sha256-kAGVJfDHEUV2qNR1LpnWq3XKBJy4hD3Swh6LX5shJpM="
                ]
        , nativeBuildInputs ["cmake", "doxygen", "graphviz"]
        , buildInputs ["gtest"]
        , checkInputs ["valgrind"]
        , cmakeFlags
            [ "-DRAPIDJSON_BUILD_DOC=ON"
            , "-DRAPIDJSON_BUILD_TESTS=ON"
            , "-DRAPIDJSON_BUILD_EXAMPLES=ON"
            , "-DRAPIDJSON_BUILD_CXX11=OFF"
            , "-DRAPIDJSON_BUILD_CXX17=ON"
            , "-DRAPIDJSON_ENABLE_INSTRUMENTATION_OPT=OFF"
            , "-DCMAKE_CXX_FLAGS_RELEASE=-Wno-error"
            ]
        , postPatch
            [ Sub.inPlace
                "doc/Doxyfile.in"
                [Sub.replace "WARN_IF_UNDOCUMENTED   = YES" "WARN_IF_UNDOCUMENTED   = NO"]
            , Sub.inPlace
                "doc/Doxyfile.zh-cn.in"
                [Sub.replace "WARN_IF_UNDOCUMENTED   = YES" "WARN_IF_UNDOCUMENTED   = NO"]
            ]
        , doCheck True
        , description "Fast JSON parser/generator for C++ with SAX/DOM style API"
        , homepage "http://rapidjson.org/"
        , license "mit"
        ]
