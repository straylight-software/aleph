{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.CMake
Description : Typed wrapper for CMake configuration

CMake is a cross-platform build system generator. This module provides
typed options for common CMake configuration flags.

@
import Aleph.Script.Tools.CMake as CMake

-- Configure for static library build
let flags = CMake.buildArgs CMake.defaults
      { CMake.buildStaticLibs = True
      , CMake.buildSharedLibs = False
      , CMake.buildType = Just Release
      }
-- flags = ["-DBUILD_STATIC_LIBS=ON", "-DBUILD_SHARED_LIBS=OFF", "-DCMAKE_BUILD_TYPE=Release"]
@
-}
module Aleph.Script.Tools.CMake (
    -- * Options
    Options (..),
    defaults,

    -- * Build types
    BuildType (..),

    -- * Argument building
    buildArgs,

    -- * Testing
    testZlibNgFlags,
    testFmtFlags,
) where

import Data.List (sort)
import Data.Maybe (catMaybes)
import Data.Text (Text)
import qualified Data.Text as T
import Prelude

-- | CMake build type
data BuildType
    = Release
    | Debug
    | RelWithDebInfo
    | MinSizeRel
    deriving (Show, Eq)

buildTypeToText :: BuildType -> Text
buildTypeToText Release = "Release"
buildTypeToText Debug = "Debug"
buildTypeToText RelWithDebInfo = "RelWithDebInfo"
buildTypeToText MinSizeRel = "MinSizeRel"

{- | CMake configuration options

Use 'defaults' and override fields as needed:

> defaults { buildStaticLibs = Just True, buildSharedLibs = Just False }

== Turing Registry Interaction

When using aleph-naught's stdenv, compiler flags come from the Turing Registry
(NIX_CFLAGS_COMPILE, hardeningDisable, etc.), NOT from CMake options.

Generally you should NOT set:

  * 'buildType' - stdenv provides flags via environment
  * 'cxxStandard' - turing-registry mandates C++23
  * 'cCompiler'/'cxxCompiler' - stdenv provides the toolchain

These options exist for compatibility with non-straylight builds or edge cases.
-}
data Options = Options
    { -- \** Install paths
      installPrefix :: Maybe Text
    -- ^ CMAKE_INSTALL_PREFIX
    , -- \** Build configuration
      buildType :: Maybe BuildType
    -- ^ CMAKE_BUILD_TYPE (usually leave Nothing for aleph-naught)
    , buildSharedLibs :: Maybe Bool
    -- ^ BUILD_SHARED_LIBS
    , buildStaticLibs :: Maybe Bool
    -- ^ BUILD_STATIC_LIBS (project-specific but common)
    , -- \** Compiler settings
      cxxStandard :: Maybe Int
    -- ^ CMAKE_CXX_STANDARD (11, 14, 17, 20, 23)
    , cStandard :: Maybe Int
    -- ^ CMAKE_C_STANDARD (99, 11, 17, 23)
    , positionIndependentCode :: Maybe Bool
    -- ^ CMAKE_POSITION_INDEPENDENT_CODE
    , -- \** Toolchain
      cCompiler :: Maybe Text
    -- ^ CMAKE_C_COMPILER
    , cxxCompiler :: Maybe Text
    -- ^ CMAKE_CXX_COMPILER
    , linker :: Maybe Text
    -- ^ CMAKE_LINKER
    , -- \** Features
      buildTesting :: Maybe Bool
    -- ^ BUILD_TESTING
    , buildExamples :: Maybe Bool
    -- ^ BUILD_EXAMPLES (project-specific but common)
    , buildDocs :: Maybe Bool
    -- ^ BUILD_DOCS (project-specific but common)
    , -- \** Escape hatch for project-specific flags
      extraFlags :: [(Text, Text)]
    -- ^ Additional -DNAME=VALUE pairs
    }
    deriving (Show, Eq)

-- | Default options - no flags set, let CMake/project use defaults
defaults :: Options
defaults =
    Options
        { installPrefix = Nothing
        , buildType = Nothing
        , buildSharedLibs = Nothing
        , buildStaticLibs = Nothing
        , cxxStandard = Nothing
        , cStandard = Nothing
        , positionIndependentCode = Nothing
        , cCompiler = Nothing
        , cxxCompiler = Nothing
        , linker = Nothing
        , buildTesting = Nothing
        , buildExamples = Nothing
        , buildDocs = Nothing
        , extraFlags = []
        }

{- | Build CMake command-line arguments from options

>>> buildArgs defaults { buildStaticLibs = Just True }
["-DBUILD_STATIC_LIBS=ON"]

>>> buildArgs defaults { buildType = Just Release, cxxStandard = Just 17 }
["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_STANDARD=17"]
-}
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ opt installPrefix "CMAKE_INSTALL_PREFIX"
        , optEnum buildType "CMAKE_BUILD_TYPE" buildTypeToText
        , optBool buildSharedLibs "BUILD_SHARED_LIBS"
        , optBool buildStaticLibs "BUILD_STATIC_LIBS"
        , optShow cxxStandard "CMAKE_CXX_STANDARD"
        , optShow cStandard "CMAKE_C_STANDARD"
        , optBool positionIndependentCode "CMAKE_POSITION_INDEPENDENT_CODE"
        , opt cCompiler "CMAKE_C_COMPILER"
        , opt cxxCompiler "CMAKE_CXX_COMPILER"
        , opt linker "CMAKE_LINKER"
        , optBool buildTesting "BUILD_TESTING"
        , optBool buildExamples "BUILD_EXAMPLES"
        , optBool buildDocs "BUILD_DOCS"
        ]
        ++ map mkExtra extraFlags
  where
    opt :: Maybe Text -> Text -> Maybe Text
    opt (Just v) name = Just ("-D" <> name <> "=" <> v)
    opt Nothing _ = Nothing

    optBool :: Maybe Bool -> Text -> Maybe Text
    optBool (Just True) name = Just ("-D" <> name <> "=ON")
    optBool (Just False) name = Just ("-D" <> name <> "=OFF")
    optBool Nothing _ = Nothing

    optShow :: (Show a) => Maybe a -> Text -> Maybe Text
    optShow (Just v) name = Just ("-D" <> name <> "=" <> T.pack (show v))
    optShow Nothing _ = Nothing

    optEnum :: Maybe a -> Text -> (a -> Text) -> Maybe Text
    optEnum (Just v) name f = Just ("-D" <> name <> "=" <> f v)
    optEnum Nothing _ _ = Nothing

    mkExtra :: (Text, Text) -> Text
    mkExtra (name, val) = "-D" <> name <> "=" <> val

-- ============================================================================
-- Tests - compare against known-good flags from libmodern-cpp
-- ============================================================================

{- | Test: zlib-ng flags match libmodern-cpp/nix/packages/zlib-ng/default.nix

Expected flags from that file:
  cmakeFlags = [
    "-DCMAKE_INSTALL_PREFIX=/"
    "-DBUILD_STATIC_LIBS=ON"
    "-DBUILD_SHARED_LIBS=OFF"
    "-DINSTALL_UTILS=ON"
    "-DZLIB_COMPAT=ON"
  ];

Note: Order doesn't matter for cmake, so we compare as sets (sorted lists).
-}
testZlibNgFlags :: Bool
testZlibNgFlags = sort (buildArgs zlibNgOpts) == sort expectedFlags
  where
    zlibNgOpts =
        defaults
            { installPrefix = Just "/"
            , buildStaticLibs = Just True
            , buildSharedLibs = Just False
            , extraFlags =
                [ ("INSTALL_UTILS", "ON")
                , ("ZLIB_COMPAT", "ON")
                ]
            }
    -- Exactly as in libmodern-cpp/nix/packages/zlib-ng/default.nix
    expectedFlags =
        [ "-DCMAKE_INSTALL_PREFIX=/"
        , "-DBUILD_STATIC_LIBS=ON"
        , "-DBUILD_SHARED_LIBS=OFF"
        , "-DINSTALL_UTILS=ON"
        , "-DZLIB_COMPAT=ON"
        ]

{- | Test: fmt flags match libmodern-cpp/nix/packages/fmt/default.nix

Expected flags from that file:
  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "RelWithDebInfo")
    (lib.cmakeFeature "CMAKE_CXX_STANDARD" "17")
    (lib.cmakeBool "CMAKE_POSITION_INDEPENDENT_CODE" true)
    (lib.cmakeBool "BUILD_STATIC_LIBS" true)
    (lib.cmakeBool "BUILD_SHARED_LIBS" false)
  ];

Note: Order doesn't matter for cmake, so we compare as sets (sorted lists).
-}
testFmtFlags :: Bool
testFmtFlags = sort (buildArgs fmtOpts) == sort expectedFlags
  where
    fmtOpts =
        defaults
            { buildType = Just RelWithDebInfo
            , cxxStandard = Just 17
            , positionIndependentCode = Just True
            , buildStaticLibs = Just True
            , buildSharedLibs = Just False
            }
    -- Exactly as in libmodern-cpp/nix/packages/fmt/default.nix (expanded from lib.cmake* helpers)
    expectedFlags =
        [ "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
        , "-DCMAKE_CXX_STANDARD=17"
        , "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
        , "-DBUILD_STATIC_LIBS=ON"
        , "-DBUILD_SHARED_LIBS=OFF"
        ]
