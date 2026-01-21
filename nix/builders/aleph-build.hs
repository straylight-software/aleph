{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

-- | aleph-build: Unified build dispatcher
--
-- Reads BuildContext.dhall and dispatches to the appropriate build logic.
-- No env vars except ALEPH_CONTEXT which points to the context file.
--
-- Usage:
--   ALEPH_CONTEXT=/path/to/context.dhall aleph-build
--
-- The build type is determined by reading the Package spec from the context.

module Main where

import Aleph.Build

import Control.Exception (SomeException, catch)
import Control.Monad (forM_)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Dhall (FromDhall, auto, input)
import GHC.Generics (Generic)
import System.Directory (listDirectory, doesDirectoryExist, doesFileExist, copyFile)
import System.Environment (getEnv, lookupEnv)
import System.FilePath ((</>))
import System.IO (hPutStrLn, stderr)

--------------------------------------------------------------------------------
-- Build Type Detection
--------------------------------------------------------------------------------

-- | Build type from the Package spec
-- We need to read this from the context file
data BuildType
  = CMakeBuild CMakeConfig
  | AutotoolsBuild AutotoolsConfig
  | MesonBuild MesonConfig
  | HeaderOnlyBuild HeaderOnlyConfig
  | CustomBuild CustomConfig
  deriving (Show, Generic)

data CMakeConfig = CMakeConfig
  { cmakeFlags :: [Text]
  , cmakeBuildType :: Text
  , cmakeLinkage :: Text
  } deriving (Show, Generic, FromDhall)

data AutotoolsConfig = AutotoolsConfig
  { configureFlags :: [Text]
  } deriving (Show, Generic, FromDhall)

data MesonConfig = MesonConfig
  { mesonFlags :: [Text]
  } deriving (Show, Generic, FromDhall)

data HeaderOnlyConfig = HeaderOnlyConfig
  { includeDir :: Text
  } deriving (Show, Generic, FromDhall)

data CustomConfig = CustomConfig
  { builderScript :: Text
  } deriving (Show, Generic, FromDhall)

--------------------------------------------------------------------------------
-- Build Dispatcher
--------------------------------------------------------------------------------

main :: IO ()
main = do
  -- Get context
  ctx <- getCtx
  log' "Building from context"
  
  -- Read the Package spec to determine build type
  -- For now, we infer from available deps and context
  -- A proper implementation would read the full spec
  buildType <- detectBuildType ctx
  
  case buildType of
    "cmake" -> buildCMake ctx
    "autotools" -> buildAutotools ctx
    "meson" -> buildMeson ctx
    "header-only" -> buildHeaderOnly ctx
    other -> failBuild $ "Unknown build type: " <> other

-- | Detect build type from context
-- This is a heuristic for now; the real solution reads the spec
detectBuildType :: Ctx -> IO String
detectBuildType ctx = do
  mBuildType <- lookupEnv "ALEPH_BUILD_TYPE"
  case mBuildType of
    Just bt -> pure bt
    Nothing -> do
      -- Heuristic: check what files exist in source
      hasCMake <- hasFile' (srcPath ctx "CMakeLists.txt")
      hasConfigure <- hasFile' (srcPath ctx "configure")
      hasMeson <- hasFile' (srcPath ctx "meson.build")
      
      if hasCMake then pure "cmake"
      else if hasMeson then pure "meson"
      else if hasConfigure then pure "autotools"
      else pure "header-only"
  where
    hasFile' :: FilePath -> IO Bool
    hasFile' p = doesFileExist p `catch` handler
      where
        handler :: SomeException -> IO Bool
        handler _ = pure False

--------------------------------------------------------------------------------
-- Build Implementations
--------------------------------------------------------------------------------

buildCMake :: Ctx -> IO ()
buildCMake ctx = do
  log' "CMake build"
  -- Get extra flags from env (for now)
  mFlags <- lookupEnv "ALEPH_CMAKE_FLAGS"
  let extraFlags = maybe [] words mFlags
  cmake ctx extraFlags
  ninja ctx

buildAutotools :: Ctx -> IO ()
buildAutotools ctx = do
  log' "Autotools build"
  mFlags <- lookupEnv "ALEPH_CONFIGURE_FLAGS"
  let extraFlags = maybe [] words mFlags
  configure ctx extraFlags
  make ctx []
  make ctx ["install"]

buildMeson :: Ctx -> IO ()
buildMeson ctx = do
  log' "Meson build"
  mFlags <- lookupEnv "ALEPH_MESON_FLAGS"
  let extraFlags = maybe [] words mFlags
  meson ctx extraFlags
  mesonCompile ctx
  mesonInstall ctx

buildHeaderOnly :: Ctx -> IO ()
buildHeaderOnly ctx = do
  log' "Header-only build"
  -- Find include directory
  includeDir <- lookupEnv "ALEPH_INCLUDE_DIR" >>= \case
    Just d -> pure d
    Nothing -> do
      -- Try common locations
      let candidates = ["include", "src", "."]
      findIncludeDir ctx candidates
  
  mkdir (outPath ctx "include")
  cp (srcPath ctx includeDir) (outPath ctx "include")

findIncludeDir :: Ctx -> [FilePath] -> IO FilePath
findIncludeDir _ [] = failBuild "Could not find include directory"
findIncludeDir ctx (d:ds) = do
  exists <- doesDirectoryExist (srcPath ctx d)
  if exists then pure d else findIncludeDir ctx ds

--------------------------------------------------------------------------------
-- Logging
--------------------------------------------------------------------------------

log' :: String -> IO ()
log' msg = hPutStrLn stderr $ "[aleph-build] " <> msg
