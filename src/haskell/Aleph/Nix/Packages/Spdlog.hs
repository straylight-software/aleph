{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}

{- | spdlog - Super fast C++ logging library

Uses external fmt library. Note the typed CMake options.

Original Nix:
@
stdenv.mkDerivation {
  pname = "spdlog";
  version = "1.15.2";

  src = fetchFromGitHub {
    owner = "gabime";
    repo = "spdlog";
    tag = "v1.15.2";
    hash = "sha256-...";
  };

  nativeBuildInputs = [ cmake ];
  buildInputs = [ fmt ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DCMAKE_CXX_STANDARD=17"
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    "-DBUILD_STATIC_LIBS=ON"
    "-DBUILD_SHARED_LIBS=OFF"
    "-DSPDLOG_FMT_EXTERNAL_HO=ON"
    "-DSPDLOG_BUILD_EXAMPLE=OFF"
    "-DSPDLOG_BUILD_BENCH=OFF"
    "-DSPDLOG_BUILD_TESTS=OFF"
  ];

  postInstall = ''
    mkdir -p $out/share/doc/spdlog
    cp -rv ../example $out/share/doc/spdlog
  '';
}
@
-}
module Aleph.Nix.Packages.Spdlog (spdlog) where

import Aleph.Nix.Derivation (Drv)
import Aleph.Nix.Syntax

spdlog :: Drv
spdlog =
    mkDerivation
        [ pname "spdlog"
        , version "1.15.2"
        , src $
            fetchFromGitHub
                [ owner "gabime"
                , repo "spdlog"
                , rev "v1.15.2"
                , hash "sha256-9RhB4GdFjZbCIfMOWWriLAUf9DE/i/+FTXczr0pD0Vg="
                ]
        , nativeBuildInputs ["cmake"]
        , buildInputs ["fmt"]
        , -- Typed CMake options - no string flags!
          cmake
            defaults
                { buildType = Just RelWithDebInfo
                , cxxStandard = Just 17
                , positionIndependentCode = Just True
                , buildStaticLibs = Just True
                , buildSharedLibs = Just False
                , buildExamples = Just False
                , buildTesting = Just False
                , extraFlags =
                    [ ("SPDLOG_FMT_EXTERNAL_HO", "ON")
                    , ("SPDLOG_BUILD_EXAMPLE", "OFF")
                    , ("SPDLOG_BUILD_BENCH", "OFF")
                    , ("SPDLOG_BUILD_TESTS", "OFF")
                    ]
                }
        , -- Copy examples to doc directory
          postInstall
            [ mkdir "share/doc/spdlog"
            , copy "../example" "share/doc/spdlog/"
            ]
        , description "Super fast C++ logging library"
        , homepage "https://github.com/gabime/spdlog"
        , license "mit"
        ]
