{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}

{- | RapidJSON - fast JSON parser/generator for C++.

Ported from s4/nix/packages/libmodern-cpp/rapidjson/default.nix

Compare the Nix:
@
stdenv.mkDerivation {
  pname = "rapidjson";
  version = "unstable-2024-04-09";

  src = fetchFromGitHub {
    owner = "Tencent";
    repo = "rapidjson";
    rev = "ab1842a2dae061284c0a62dca1cc6d5e7e37e346";
    hash = "sha256-kAGVJfDHEUV2qNR1LpnWq3XKBJy4hD3Swh6LX5shJpM=";
  };

  nativeBuildInputs = [ cmake doxygen graphviz ];
  buildInputs = [ gtest ];

  cmakeFlags = [
    (lib.cmakeBool "RAPIDJSON_BUILD_DOC" true)
    (lib.cmakeBool "RAPIDJSON_BUILD_TESTS" true)
    (lib.cmakeBool "RAPIDJSON_BUILD_EXAMPLES" true)
    (lib.cmakeBool "RAPIDJSON_BUILD_CXX11" false)
    (lib.cmakeBool "RAPIDJSON_BUILD_CXX17" true)
    (lib.cmakeBool "RAPIDJSON_ENABLE_INSTRUMENTATION_OPT" false)
    (lib.cmakeFeature "CMAKE_CXX_FLAGS_RELEASE" "-Wno-error")
  ];

  postPatch = \'\'
    for f in doc/Doxyfile.*; do
      substituteInPlace $f --replace-fail "WARN_IF_UNDOCUMENTED   = YES" "WARN_IF_UNDOCUMENTED   = NO"
    done
  \'\';

  doCheck = true;
  nativeCheckInputs = [ valgrind ];
}
@
-}
module Aleph.Nix.Packages.Rapidjson (rapidjson) where

import Aleph.Nix.Derivation (Drv)
import Aleph.Nix.Syntax

rapidjson :: Drv
rapidjson =
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
            [ substitute
                "doc/Doxyfile.in"
                [("WARN_IF_UNDOCUMENTED   = YES", "WARN_IF_UNDOCUMENTED   = NO")]
            , substitute
                "doc/Doxyfile.zh-cn.in"
                [("WARN_IF_UNDOCUMENTED   = YES", "WARN_IF_UNDOCUMENTED   = NO")]
            ]
        , doCheck True
        , description "Fast JSON parser/generator for C++ with SAX/DOM style API"
        , homepage "http://rapidjson.org/"
        , license "mit"
        ]
