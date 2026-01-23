{-# LANGUAGE OverloadedStrings #-}

{- | zlib-ng package definition.

Uses typed CMake options. Compare to raw string flags:

@
-- OLD: raw strings, typos invisible
cmakeFlags = ["-DBUILD_STATIC_LIBS=ON", "-DBUILD_SHARED_LIBS=OFF"]

-- NEW: typed, compiler-checked
cmake defaults
    { buildStaticLibs = Just True
    , buildSharedLibs = Just False
    }
@
-}
module Aleph.Nix.Packages.ZlibNg (
    zlibNg,
    zlibNgVersion,
) where

import Aleph.Nix.Derivation (Drv)
import Aleph.Nix.Syntax
import Data.Text (Text)

-- | Current version of zlib-ng we package.
zlibNgVersion :: Text
zlibNgVersion = "2.2.4"

{- | zlib-ng: zlib data compression library for next generation systems.

Features:
  - SIMD optimizations (SSE2, AVX2, NEON, etc.)
  - Modern C implementation
  - API compatible with zlib (via ZLIB_COMPAT)
-}
zlibNg :: Drv
zlibNg =
    mkDerivation
        [ pname "zlib-ng"
        , version zlibNgVersion
        , src $
            fetchFromGitHub
                [ owner "zlib-ng"
                , repo "zlib-ng"
                , rev zlibNgVersion
                , hash "sha256-Khmrhp5qy4vvoQe4WgoogpjWrgcUB/q8zZeqIydthYg="
                ]
        , nativeBuildInputs ["cmake", "pkg-config"]
        , buildInputs ["gtest"]
        , -- Typed CMake options
          cmake
            defaults
                { installPrefix = Just "/"
                , buildStaticLibs = Just True
                , buildSharedLibs = Just False
                , extraFlags =
                    [ ("INSTALL_UTILS", "ON")
                    , ("ZLIB_COMPAT", "ON") -- API compatibility with zlib
                    ]
                }
        , description "zlib data compression library for the next generation systems"
        , homepage "https://github.com/zlib-ng/zlib-ng"
        , license "zlib"
        ]
