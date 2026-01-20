{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}

{- | nlohmann/json - JSON for Modern C++

Header-only JSON library. We fix the broken pkg-config that CMake generates
and add a compatibility symlink.

Original Nix:
@
stdenv.mkDerivation {
  pname = "nlohmann_json";
  version = "3.12.0";

  src = fetchFromGitHub {
    owner = "nlohmann";
    repo = "json";
    rev = "v3.12.0";
    hash = "sha256-cECvDOLxgX7Q9R3IE86Hj9JJUxraDQvhoyPDF03B2CY=";
  };

  nativeBuildInputs = [ cmake pkg-config ];
  cmakeFlags = [ "-DJSON_BuildTests=OFF" ];

  postInstall = ''
    rm -f $out/share/pkgconfig/nlohmann_json.pc
    mkdir -p $out/lib/pkgconfig
    cat > $out/lib/pkgconfig/nlohmann_json.pc <<EOF
    ...
    EOF
    ln -s nlohmann/json.hpp $out/include/json.hpp
  '';
}
@
-}
module Aleph.Nix.Packages.NlohmannJson (nlohmannJson) where

import Aleph.Nix.Derivation (Drv)
import Aleph.Nix.Syntax
import Data.Text (Text)
import Prelude hiding (writeFile)

nlohmannJson :: Drv
nlohmannJson =
    mkDerivation
        [ pname "nlohmann_json"
        , version nlohmannJsonVersion
        , src $
            fetchFromGitHub
                [ owner "nlohmann"
                , repo "json"
                , rev ("v" <> nlohmannJsonVersion)
                , hash "sha256-cECvDOLxgX7Q9R3IE86Hj9JJUxraDQvhoyPDF03B2CY="
                ]
        , nativeBuildInputs ["cmake", "pkg-config"]
        , -- Typed CMake options
          cmake
            defaults
                { buildTesting = Just False
                , extraFlags = [("JSON_BuildTests", "OFF")]
                }
        , -- Fix the broken pkg-config and add compatibility symlink
          postInstall
            [ remove "share/pkgconfig/nlohmann_json.pc"
            , mkdir "lib/pkgconfig"
            , writeFile "lib/pkgconfig/nlohmann_json.pc" pkgConfigContent
            , symlink "nlohmann/json.hpp" "include/json.hpp"
            ]
        , description "JSON for Modern C++"
        , homepage "https://github.com/nlohmann/json"
        , license "mit"
        ]

nlohmannJsonVersion :: Text
nlohmannJsonVersion = "3.12.0"

{- | pkg-config content for nlohmann_json
Note: $out is expanded at build time by the Nix interpreter
-}
pkgConfigContent :: Text
pkgConfigContent =
    "prefix=${out}\n\
    \includedir=${prefix}/include\n\
    \\n\
    \Name: nlohmann_json\n\
    \Description: JSON for Modern C++\n\
    \Version: "
        <> nlohmannJsonVersion
        <> "\n\
           \Cflags: -I${includedir}\n"
