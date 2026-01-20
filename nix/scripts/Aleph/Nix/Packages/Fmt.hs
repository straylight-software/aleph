{-# LANGUAGE OverloadedStrings #-}

{- | fmt (fmtlib) package definition.

A fast, safe formatting library for C++.

Uses typed CMake options instead of raw flag strings.
-}
module Aleph.Nix.Packages.Fmt (
    fmt,
    fmtVersion,
) where

import Aleph.Nix.Derivation (Drv)
import Aleph.Nix.Syntax
import Data.Text (Text)

-- | Current version of fmt we package.
fmtVersion :: Text
fmtVersion = "11.2.0"

{- | fmt: Small, safe and fast formatting library.

Features:
  - Type-safe printf replacement
  - Python-like format strings
  - Compile-time format string checking (C++20)
-}
fmt :: Drv
fmt =
    mkDerivation
        [ pname "fmt"
        , version fmtVersion
        , src $
            fetchFromGitHub
                [ owner "fmtlib"
                , repo "fmt"
                , rev fmtVersion
                , hash "sha256-sAlU5L/olxQUYcv8euVYWTTB8TrVeQgXLHtXy8IMEnU="
                ]
        , nativeBuildInputs ["cmake"]
        , -- Typed CMake options instead of raw strings
          cmake
            defaults
                { buildType = Just RelWithDebInfo
                , cxxStandard = Just 17
                , positionIndependentCode = Just True
                , buildStaticLibs = Just True
                , buildSharedLibs = Just False
                }
        , -- TODO: add env support to Syntax
          -- , env [("CXXFLAGS", "-DFMT_USE_BITINT=0")]

          description "Small, safe and fast formatting library"
        , homepage "https://fmt.dev/"
        , license "mit"
        ]
