{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

{- | aleph: The Aleph-1 builder

This is the main entry point for building packages. It:

1. Reads the Dhall spec (Package.dhall)
2. Calls Nix via FFI to fetch sources
3. Calls Nix via FFI to resolve dependencies
4. Runs the build (cmake, autotools, meson, etc.)

= Control Flow

Haskell owns everything. Nix is just a library for fetching and caching.

@
  Dhall spec
      │
      ▼
  aleph (Haskell)
      │
      ├──► Aleph.Nix.Fetch.fetchGitHub  ──► [Nix via WASI FFI]
      │                                          │
      │                                          ▼
      │                                    /nix/store/...-source
      │
      ├──► Aleph.Nix.Store.resolveDep   ──► [Nix via WASI FFI]
      │                                          │
      │                                          ▼
      │                                    /nix/store/...-cmake
      │
      └──► Aleph.Build.cmake/ninja      ──► build output
@

= Usage

Compiled to WASM and invoked by the Nix derivation:

@
derivation {
  builder = "${aleph-wasm}/bin/aleph.wasm";
  args = [ "--spec" "${spec.dhall}" ];
}
@

Or for testing, compiled natively:

@
ALEPH_SPEC=./packages-dhall/zlib-ng.dhall aleph
@
-}
module Main where

import Aleph.Build
import Aleph.Nix (fetchGitHub, fetchUrl, resolveDep, getOutPath, getCores, getSystem)

import Control.Monad (forM)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Dhall (FromDhall, auto, input)
import GHC.Generics (Generic)
import GHC.Natural (Natural)
import System.Environment (getArgs, lookupEnv)
import System.IO (hPutStrLn, stderr)

--------------------------------------------------------------------------------
-- Dhall Types (matching nix/Drv/)
--------------------------------------------------------------------------------

-- | Source specification
data Src
  = SrcGitHub { owner :: Text, repo :: Text, rev :: Text, hash :: Text }
  | SrcUrl { url :: Text, hash :: Text }
  | SrcLocal { path :: Text }
  | SrcNone
  deriving (Show, Generic, FromDhall)

-- | Build system configuration
data CMakeConfig = CMakeConfig
  { flags :: [Text]
  , buildType :: Text
  , linkage :: Text
  , pic :: Text
  , lto :: Text
  } deriving (Show, Generic, FromDhall)

data AutotoolsConfig = AutotoolsConfig
  { configureFlags :: [Text]
  , makeFlags :: [Text]
  } deriving (Show, Generic, FromDhall)

data MesonConfig = MesonConfig
  { mesonFlags :: [Text]
  , mesonBuildType :: Text
  } deriving (Show, Generic, FromDhall)

data HeaderOnlyConfig = HeaderOnlyConfig
  { include :: Text
  } deriving (Show, Generic, FromDhall)

data CustomConfig = CustomConfig
  { builder :: Text
  } deriving (Show, Generic, FromDhall)

data Build
  = CMake CMakeConfig
  | Autotools AutotoolsConfig
  | Meson MesonConfig
  | HeaderOnly HeaderOnlyConfig
  | Custom CustomConfig
  deriving (Show, Generic, FromDhall)

-- | Triple (simplified - full version in Aleph.Build.Triple)
data SpecTriple = SpecTriple
  { arch :: Text
  , vendor :: Text
  , os :: Text
  , abi :: Text
  } deriving (Show, Generic, FromDhall)

-- | Package specification
data Package = Package
  { name :: Text
  , version :: Text
  , src :: Src
  , deps :: [Text]
  , build :: Build
  , host :: SpecTriple
  , target :: Maybe SpecTriple
  , checks :: [Text]
  } deriving (Show, Generic, FromDhall)

--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

main :: IO ()
main = do
  args <- getArgs
  
  -- Get spec file from args or env
  specFile <- case args of
    ["--spec", f] -> pure f
    _ -> do
      mSpec <- lookupEnv "ALEPH_SPEC"
      case mSpec of
        Just f -> pure f
        Nothing -> failBuild "Usage: aleph --spec <file.dhall> or set ALEPH_SPEC"
  
  log' $ "Reading spec: " <> specFile
  
  -- Read and parse Dhall spec
  specText <- TIO.readFile specFile
  pkg <- input auto specText
  
  log' $ "Building " <> T.unpack (name pkg) <> "-" <> T.unpack (version pkg)
  
  -- Fetch source via Nix FFI
  srcPath <- fetchSource (src pkg)
  log' $ "Source: " <> T.unpack srcPath
  
  -- Resolve dependencies via Nix FFI
  depPaths <- resolveDeps (deps pkg)
  log' $ "Resolved " <> show (length depPaths) <> " dependencies"
  
  -- Get output path and cores from Nix
  outPath <- getOutPath "out"
  cores <- getCores
  
  log' $ "Output: " <> T.unpack outPath
  log' $ "Cores: " <> show cores
  
  -- Build context
  let ctx = Ctx
        { ctxOut = T.unpack outPath
        , ctxSrc = T.unpack srcPath
        , ctxDeps = Map.fromList [(T.unpack k, T.unpack v) | (k, v) <- depPaths]
        , ctxHost = x86_64_linux_gnu  -- TODO: parse from spec
        , ctxTarget = Nothing
        , ctxCores = cores
        }
  
  -- Dispatch based on build type
  case build pkg of
    CMake cfg -> buildCMake ctx cfg
    Autotools cfg -> buildAutotools ctx cfg
    Meson cfg -> buildMeson ctx cfg
    HeaderOnly cfg -> buildHeaderOnly ctx cfg
    Custom cfg -> buildCustom ctx cfg
  
  log' "Build complete"

--------------------------------------------------------------------------------
-- Source Fetching
--------------------------------------------------------------------------------

fetchSource :: Src -> IO Text
fetchSource = \case
  SrcGitHub{..} -> fetchGitHub owner repo rev hash
  SrcUrl{..} -> fetchUrl url hash
  SrcLocal{..} -> pure path  -- Already a path
  SrcNone -> pure ""

--------------------------------------------------------------------------------
-- Dependency Resolution
--------------------------------------------------------------------------------

resolveDeps :: [Text] -> IO [(Text, Text)]
resolveDeps names = forM names $ \n -> do
  path <- resolveDep n
  pure (n, path)

--------------------------------------------------------------------------------
-- Build Implementations
--------------------------------------------------------------------------------

buildCMake :: Ctx -> CMakeConfig -> IO ()
buildCMake ctx CMakeConfig{..} = do
  log' "CMake build"
  let extraFlags = map T.unpack flags
  cmake ctx extraFlags
  ninja ctx

buildAutotools :: Ctx -> AutotoolsConfig -> IO ()
buildAutotools ctx AutotoolsConfig{..} = do
  log' "Autotools build"
  let extraFlags = map T.unpack configureFlags
  configure ctx extraFlags
  make ctx []
  make ctx ["install"]

buildMeson :: Ctx -> MesonConfig -> IO ()
buildMeson ctx MesonConfig{..} = do
  log' "Meson build"
  let extraFlags = map T.unpack mesonFlags
  meson ctx extraFlags
  mesonCompile ctx
  mesonInstall ctx

buildHeaderOnly :: Ctx -> HeaderOnlyConfig -> IO ()
buildHeaderOnly ctx HeaderOnlyConfig{..} = do
  log' "Header-only build"
  mkdir (outPath ctx "include")
  cp (srcPath ctx (T.unpack include)) (outPath ctx "include")

buildCustom :: Ctx -> CustomConfig -> IO ()
buildCustom _ctx CustomConfig{..} = do
  log' $ "Custom build: " <> T.unpack builder
  -- TODO: Invoke custom builder script
  failBuild "Custom builders not yet implemented"

--------------------------------------------------------------------------------
-- Logging
--------------------------------------------------------------------------------

log' :: String -> IO ()
log' msg = hPutStrLn stderr $ "[aleph] " <> msg
