{-# LANGUAGE OverloadedStrings #-}

-- | nlohmann/json - JSON for Modern C++
module Pkg where

import Aleph.Nix.Package
import Prelude hiding (writeFile)

pkg :: Drv
pkg =
    mkDerivation
        [ pname "nlohmann_json"
        , version "3.12.0"
        , src $
            fetchFromGitHub
                [ owner "nlohmann"
                , repo "json"
                , rev "v3.12.0"
                , hash "sha256-cECvDOLxgX7Q9R3IE86Hj9JJUxraDQvhoyPDF03B2CY="
                ]
        , nativeBuildInputs ["cmake", "pkg-config"]
        , cmake
            defaults
                { buildTesting = Just False
                , extraFlags = [("JSON_BuildTests", "OFF")]
                }
        , -- Fix broken pkg-config and add compatibility symlink
          postInstall
            [ remove "share/pkgconfig/nlohmann_json.pc"
            , mkdir "lib/pkgconfig"
            , writeFile "lib/pkgconfig/nlohmann_json.pc" pkgConfig
            , symlink "nlohmann/json.hpp" "include/json.hpp"
            ]
        , description "JSON for Modern C++"
        , homepage "https://github.com/nlohmann/json"
        , license "mit"
        ]
  where
    pkgConfig =
        "prefix=${out}\n\
        \includedir=${prefix}/include\n\
        \\n\
        \Name: nlohmann_json\n\
        \Description: JSON for Modern C++\n\
        \Version: 3.12.0\n\
        \Cflags: -I${includedir}\n"
