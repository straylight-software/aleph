{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | CMake integration for DrvSpec

Type-safe CMake configuration options for package definitions.

@
import Aleph.Nix.DrvSpec
import qualified Aleph.Nix.CMake as CMake

pkg :: DrvSpec
pkg = defaultDrvSpec
    { pname = "zlib-ng"
    , phases = emptyPhases
        { configure = [CMake.configure CMake.defaults
            { CMake.buildStaticLibs = Just True
            , CMake.buildSharedLibs = Just False
            , CMake.extraFlags = [("ZLIB_COMPAT", "ON")]
            } Ninja]
        , build = [CMake.build]
        , install = [CMake.install]
        }
    }
@
-}
module Aleph.Nix.CMake (
    -- * CMake options
    Options (..),
    defaults,
    BuildType (..),

    -- * Argument building
    buildArgs,

    -- * DrvSpec action builders
    configureAction,
    buildAction,
    installAction,

    -- * Convenience: full CMake phases
    cmakePhases,
) where

import Aleph.Nix.DrvSpec (Action (..), Generator (..), Phases (..), Ref (..), emptyPhases)
import Data.Maybe (catMaybes)
import Data.Text (Text)
import qualified Data.Text as T

-- ============================================================================
-- CMake Build Types
-- ============================================================================

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

-- ============================================================================
-- CMake Options
-- ============================================================================

{- | CMake configuration options

Use 'defaults' and override fields as needed:

> defaults { buildStaticLibs = Just True, buildSharedLibs = Just False }
-}
data Options = Options
    { buildType :: Maybe BuildType
    -- ^ CMAKE_BUILD_TYPE
    , buildSharedLibs :: Maybe Bool
    -- ^ BUILD_SHARED_LIBS
    , buildStaticLibs :: Maybe Bool
    -- ^ BUILD_STATIC_LIBS (project-specific but common)
    , cxxStandard :: Maybe Int
    -- ^ CMAKE_CXX_STANDARD (11, 14, 17, 20, 23)
    , cStandard :: Maybe Int
    -- ^ CMAKE_C_STANDARD (99, 11, 17, 23)
    , positionIndependentCode :: Maybe Bool
    -- ^ CMAKE_POSITION_INDEPENDENT_CODE
    , buildTesting :: Maybe Bool
    -- ^ BUILD_TESTING
    , buildExamples :: Maybe Bool
    -- ^ BUILD_EXAMPLES (project-specific but common)
    , buildDocs :: Maybe Bool
    -- ^ BUILD_DOCS (project-specific but common)
    , extraFlags :: [(Text, Text)]
    -- ^ Additional -DNAME=VALUE pairs
    }
    deriving (Show, Eq)

-- | Default options - no flags set, let CMake/project use defaults
defaults :: Options
defaults =
    Options
        { buildType = Nothing
        , buildSharedLibs = Nothing
        , buildStaticLibs = Nothing
        , cxxStandard = Nothing
        , cStandard = Nothing
        , positionIndependentCode = Nothing
        , buildTesting = Nothing
        , buildExamples = Nothing
        , buildDocs = Nothing
        , extraFlags = []
        }

-- ============================================================================
-- Argument Building
-- ============================================================================

{- | Build CMake command-line arguments from options

>>> buildArgs defaults { buildStaticLibs = Just True }
["-DBUILD_STATIC_LIBS=ON"]
-}
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ optBool buildSharedLibs "BUILD_SHARED_LIBS"
        , optBool buildStaticLibs "BUILD_STATIC_LIBS"
        , optShow cxxStandard "CMAKE_CXX_STANDARD"
        , optShow cStandard "CMAKE_C_STANDARD"
        , optBool positionIndependentCode "CMAKE_POSITION_INDEPENDENT_CODE"
        , optBool buildTesting "BUILD_TESTING"
        , optBool buildExamples "BUILD_EXAMPLES"
        , optBool buildDocs "BUILD_DOCS"
        ]
        ++ map mkExtra extraFlags
  where
    optBool :: Maybe Bool -> Text -> Maybe Text
    optBool (Just True) name = Just ("-D" <> name <> "=ON")
    optBool (Just False) name = Just ("-D" <> name <> "=OFF")
    optBool Nothing _ = Nothing

    optShow :: (Show a) => Maybe a -> Text -> Maybe Text
    optShow (Just v) name = Just ("-D" <> name <> "=" <> T.pack (show v))
    optShow Nothing _ = Nothing

    mkExtra :: (Text, Text) -> Text
    mkExtra (name, val) = "-D" <> name <> "=" <> val

-- ============================================================================
-- DrvSpec Action Builders
-- ============================================================================

-- | Reference to source directory
src :: Ref
src = RefSrc Nothing

-- | Reference to output directory
out :: Ref
out = RefOut "out" Nothing

{- | Create a CMakeConfigure action from typed options

The install prefix is automatically set to $out.
Source directory defaults to $src.
Build directory is "build" relative to working directory.
-}
configureAction :: Options -> Generator -> Action
configureAction opts gen =
    CMakeConfigure
        { cmakeSrcDir = src
        , cmakeBuildDir = RefRel "build"
        , cmakeInstallPrefix = out
        , cmakeBuildType = maybe "Release" buildTypeToText (buildType opts)
        , cmakeFlags = buildArgs opts
        , cmakeGenerator = gen
        }

-- | CMakeBuild action with default settings (build in "build" directory)
buildAction :: Action
buildAction =
    CMakeBuild
        { cmakeBuildBuildDir = RefRel "build"
        , cmakeBuildTarget = Nothing
        , cmakeBuildJobs = Nothing
        }

-- | CMakeInstall action (install from "build" directory)
installAction :: Action
installAction =
    CMakeInstall
        { cmakeInstallBuildDir = RefRel "build"
        }

-- | Create complete CMake phases from options
cmakePhases :: Options -> Generator -> Phases
cmakePhases opts gen =
    emptyPhases
        { configure = [configureAction opts gen]
        , build = [buildAction]
        , install = [installAction]
        }
