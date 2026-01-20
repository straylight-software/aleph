{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}

{- | Catch2 - Modern C++ test framework

Simple CMake package with typed options.

Original Nix:
@
stdenv.mkDerivation {
  pname = "catch2";
  version = "3.3.2";

  src = fetchFromGitHub {
    owner = "catchorg";
    repo = "Catch2";
    rev = "v3.3.2";
    sha256 = "sha256-t/4iCrzPeDZNNlgibVqx5rhe+d3lXwm1GmBMDDId0VQ=";
  };

  nativeBuildInputs = [ cmake ninja ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DCMAKE_CXX_STANDARD=17"
    "-DBUILD_STATIC_LIBS=ON"
    "-DBUILD_SHARED_LIBS=OFF"
  ];
}
@
-}
module Aleph.Nix.Packages.Catch2 (catch2) where

import Aleph.Nix.Derivation (Drv)
import Aleph.Nix.Syntax

catch2 :: Drv
catch2 =
    mkDerivation
        [ pname "catch2"
        , version "3.3.2"
        , src $
            fetchFromGitHub
                [ owner "catchorg"
                , repo "Catch2"
                , rev "v3.3.2"
                , hash "sha256-t/4iCrzPeDZNNlgibVqx5rhe+d3lXwm1GmBMDDId0VQ="
                ]
        , nativeBuildInputs ["cmake", "ninja"]
        , -- Typed CMake options
          cmake
            defaults
                { buildType = Just RelWithDebInfo
                , cxxStandard = Just 17
                , buildStaticLibs = Just True
                , buildSharedLibs = Just False
                }
        , description "Modern C++ test framework"
        , homepage "https://github.com/catchorg/Catch2"
        , license "bsl-1.0"
        ]
