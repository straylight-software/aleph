{-# LANGUAGE OverloadedStrings #-}

-- | spdlog - Super fast C++ logging library
module Pkg where

import Aleph.Nix.Package

pkg :: Drv
pkg =
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
        , cmake
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
        , postInstall
            [ mkdir "share/doc/spdlog"
            , copy "../example" "share/doc/spdlog/"
            ]
        , description "Super fast C++ logging library"
        , homepage "https://github.com/gabime/spdlog"
        , license "mit"
        ]
