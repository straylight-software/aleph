{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

-- | Aleph.Build
-- The library. Every builder imports this.
module Aleph.Build
  ( -- Re-exports
    module Aleph.Build.Flags
  , module Aleph.Build.Triple

    -- Context
  , Ctx (..)
  , getCtx

    -- Paths
  , out
  , src
  , dep
  , outPath
  , srcPath
  , depPath

    -- Filesystem
  , cp
  , mv
  , ln
  , mkdir
  , rm
  , chmod
  , write
  , glob

    -- Exec
  , run
  , run_
  , runIn

    -- Build systems
  , cmake
  , ninja
  , make
  , configure
  , meson
  , mesonCompile
  , mesonInstall

    -- pkg-config
  , pkgConfig
  , pkgConfigCflags
  , pkgConfigLibs

    -- Cross compilation
  , Toolchain (..)
  , toolchain
  , crossCMakeFlags
  , crossAutotoolsFlags

    -- Checks
  , check
  , failBuild
  , noSharedLibs
  , hasFile
  , hasDir
  ) where

import Aleph.Build.Flags
import Aleph.Build.Triple

import Control.Monad (forM_, unless, when, void)
import Data.List (intercalate)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe)
import System.Directory
import System.Environment (getEnv, lookupEnv)
import System.Exit (ExitCode (..), exitFailure)
import System.FilePath ((</>), takeExtension, takeDirectory)
import System.IO (hPutStrLn, stderr)
import System.Posix.Files (createSymbolicLink, setFileMode)
import System.Process (callProcess, readProcess, readProcessWithExitCode, spawnProcess, waitForProcess)

--------------------------------------------------------------------------------
-- Context
--------------------------------------------------------------------------------

-- | Build context - everything a builder needs
data Ctx = Ctx
  { ctxOut :: FilePath
  , ctxSrc :: FilePath
  , ctxDeps :: Map String FilePath
  , ctxHost :: Triple
  , ctxTarget :: Maybe Triple
  , ctxCores :: Int
  }
  deriving (Show)

-- | Get build context from environment
getCtx :: IO Ctx
getCtx = do
  ctxOut <- getEnv "out"
  ctxSrc <- fromMaybe "." <$> lookupEnv "src"
  ctxDeps <- parseDeps . fromMaybe "" <$> lookupEnv "ALEPH_DEPS"
  ctxHost <- parseTripleEnv "ALEPH_HOST"
  ctxTarget <- do
    mt <- lookupEnv "ALEPH_TARGET"
    case mt of
      Nothing -> pure Nothing
      Just "" -> pure Nothing
      Just t -> case parse t of
        Just triple -> pure (Just triple)
        Nothing -> error $ "Invalid target triple: " <> t
  ctxCores <- maybe 1 read <$> lookupEnv "NIX_BUILD_CORES"
  pure Ctx {..}
  where
    parseDeps :: String -> Map String FilePath
    parseDeps "" = Map.empty
    parseDeps s = Map.fromList $ map parsePair $ filter (not . null) $ splitOn ':' s
      where
        parsePair p = case break (== '=') p of
          (name, '=' : path) -> (name, path)
          _ -> error $ "Invalid dep format: " <> p

    splitOn :: Char -> String -> [String]
    splitOn _ "" = []
    splitOn c s = case break (== c) s of
      (x, "") -> [x]
      (x, _ : rest) -> x : splitOn c rest

    parseTripleEnv :: String -> IO Triple
    parseTripleEnv var = do
      ms <- lookupEnv var
      case ms of
        Nothing -> pure x86_64_linux_gnu  -- default
        Just s -> case parse s of
          Just t -> pure t
          Nothing -> error $ "Invalid triple in " <> var <> ": " <> s

--------------------------------------------------------------------------------
-- Paths
--------------------------------------------------------------------------------

out :: Ctx -> FilePath
out = ctxOut

src :: Ctx -> FilePath
src = ctxSrc

dep :: Ctx -> String -> FilePath
dep ctx name = case Map.lookup name (ctxDeps ctx) of
  Just p -> p
  Nothing -> error $ "Unknown dependency: " <> name

outPath :: Ctx -> FilePath -> FilePath
outPath ctx p = ctxOut ctx </> p

srcPath :: Ctx -> FilePath -> FilePath
srcPath ctx p = ctxSrc ctx </> p

depPath :: Ctx -> String -> FilePath -> FilePath
depPath ctx name p = dep ctx name </> p

--------------------------------------------------------------------------------
-- Filesystem operations
--------------------------------------------------------------------------------

-- | Copy file or directory
cp :: FilePath -> FilePath -> IO ()
cp s d = do
  createDirectoryIfMissing True (takeDirectory d)
  isDir <- doesDirectoryExist s
  if isDir
    then copyDirectoryRecursive s d
    else copyFile s d
  where
    copyDirectoryRecursive :: FilePath -> FilePath -> IO ()
    copyDirectoryRecursive from to = do
      createDirectoryIfMissing True to
      entries <- listDirectory from
      forM_ entries $ \entry -> do
        let fromPath = from </> entry
            toPath = to </> entry
        isD <- doesDirectoryExist fromPath
        if isD
          then copyDirectoryRecursive fromPath toPath
          else copyFile fromPath toPath

    takeDirectory :: FilePath -> FilePath
    takeDirectory = reverse . dropWhile (/= '/') . reverse

-- | Move file or directory
mv :: FilePath -> FilePath -> IO ()
mv s d = do
  createDirectoryIfMissing True (takeDirectory d)
  renamePath s d
  where
    takeDirectory = reverse . dropWhile (/= '/') . reverse

-- | Create symbolic link
ln :: FilePath -> FilePath -> IO ()
ln target link = do
  createDirectoryIfMissing True (takeDirectory link)
  createSymbolicLink target link
  where
    takeDirectory = reverse . dropWhile (/= '/') . reverse

-- | Create directory
mkdir :: FilePath -> IO ()
mkdir = createDirectoryIfMissing True

-- | Remove file or directory
rm :: FilePath -> IO ()
rm = removePathForcibly

-- | Set file mode
chmod :: FilePath -> Int -> IO ()
chmod path mode = setFileMode path (fromIntegral mode)

-- | Write file
write :: FilePath -> String -> IO ()
write path content = do
  createDirectoryIfMissing True (takeDirectory path)
  writeFile path content
  where
    takeDirectory = reverse . dropWhile (/= '/') . reverse

-- | Glob files (simplified - just list directory for now)
glob :: FilePath -> IO [FilePath]
glob pattern = do
  -- Simple implementation: list directory matching pattern suffix
  let dir = takeDirectory pattern
  exists <- doesDirectoryExist dir
  if exists
    then do
      entries <- listDirectory dir
      pure $ map (dir </>) entries
    else pure []

--------------------------------------------------------------------------------
-- Exec
--------------------------------------------------------------------------------

-- | Run command, fail on error
run :: FilePath -> [String] -> IO ()
run bin args = do
  log' $ bin <> " " <> unwords args
  code <- spawnProcess bin args >>= waitForProcess
  case code of
    ExitSuccess -> pure ()
    ExitFailure n -> failBuild $ bin <> " failed with exit code " <> show n

-- | Run command, ignore exit code
run_ :: FilePath -> [String] -> IO ()
run_ bin args = do
  log' $ bin <> " " <> unwords args
  _ <- spawnProcess bin args >>= waitForProcess
  pure ()

-- | Run command in directory
runIn :: FilePath -> FilePath -> [String] -> IO ()
runIn dir bin args = do
  log' $ "cd " <> dir <> " && " <> bin <> " " <> unwords args
  cwd <- getCurrentDirectory
  setCurrentDirectory dir
  run bin args
  setCurrentDirectory cwd

--------------------------------------------------------------------------------
-- Build systems
--------------------------------------------------------------------------------

-- | Run CMake configure
cmake :: Ctx -> [String] -> IO ()
cmake ctx extraFlags = do
  let buildDir = out ctx </> "build"
      tc = toolchain ctx
      flags =
        [ "-S"
        , src ctx
        , "-B"
        , buildDir
        , "-DCMAKE_INSTALL_PREFIX=" <> out ctx
        , "-DCMAKE_BUILD_TYPE=Release"
        , "-GNinja"
        ]
          <> crossCMakeFlags tc
          <> extraFlags
  mkdir buildDir
  run (depPath ctx "cmake" "bin/cmake") flags

-- | Run ninja build and install
ninja :: Ctx -> IO ()
ninja ctx = do
  let buildDir = out ctx </> "build"
      ninjaPath = depPath ctx "ninja" "bin/ninja"
      jobs = ["-j" <> show (ctxCores ctx)]
  run ninjaPath $ ["-C", buildDir] <> jobs
  run ninjaPath ["-C", buildDir, "install"]

-- | Run make
make :: Ctx -> [String] -> IO ()
make ctx targets = do
  let makePath = depPath ctx "gnumake" "bin/make"
      jobs = ["-j" <> show (ctxCores ctx)]
  run makePath $ jobs <> targets

-- | Run autotools configure
configure :: Ctx -> [String] -> IO ()
configure ctx extraFlags = do
  let tc = toolchain ctx
      flags =
        ["--prefix=" <> out ctx]
          <> crossAutotoolsFlags tc
          <> extraFlags
  runIn (src ctx) "./configure" flags

-- | Run meson setup
meson :: Ctx -> [String] -> IO ()
meson ctx extraFlags = do
  let buildDir = out ctx </> "build"
      flags =
        [ "--prefix=" <> out ctx
        , "--buildtype=release"
        , src ctx
        , buildDir
        ]
          <> extraFlags
  mkdir buildDir
  run (depPath ctx "meson" "bin/meson") $ ["setup"] <> flags

-- | Run meson compile
mesonCompile :: Ctx -> IO ()
mesonCompile ctx = do
  let buildDir = out ctx </> "build"
  run (depPath ctx "meson" "bin/meson") ["compile", "-C", buildDir]

-- | Run meson install
mesonInstall :: Ctx -> IO ()
mesonInstall ctx = do
  let buildDir = out ctx </> "build"
  run (depPath ctx "meson" "bin/meson") ["install", "-C", buildDir]

--------------------------------------------------------------------------------
-- pkg-config
--------------------------------------------------------------------------------

-- | Get pkg-config flags
pkgConfig :: Ctx -> String -> IO [String]
pkgConfig ctx pkg = do
  let pc = depPath ctx "pkg-config" "bin/pkg-config"
  out <- readProcess pc ["--cflags", "--libs", pkg] ""
  pure $ words out

-- | Get pkg-config cflags only
pkgConfigCflags :: Ctx -> String -> IO [String]
pkgConfigCflags ctx pkg = do
  let pc = depPath ctx "pkg-config" "bin/pkg-config"
  out <- readProcess pc ["--cflags", pkg] ""
  pure $ words out

-- | Get pkg-config libs only
pkgConfigLibs :: Ctx -> String -> IO [String]
pkgConfigLibs ctx pkg = do
  let pc = depPath ctx "pkg-config" "bin/pkg-config"
  out <- readProcess pc ["--libs", pkg] ""
  pure $ words out

--------------------------------------------------------------------------------
-- Cross compilation
--------------------------------------------------------------------------------

-- | Toolchain paths
data Toolchain = Toolchain
  { tcCC :: FilePath
  , tcCXX :: FilePath
  , tcAR :: FilePath
  , tcLD :: FilePath
  , tcStrip :: FilePath
  , tcTriple :: Maybe Triple
  }
  deriving (Show)

-- | Get toolchain for context
toolchain :: Ctx -> Toolchain
toolchain ctx = case ctxTarget ctx of
  Nothing -> hostToolchain ctx
  Just t -> crossToolchain ctx t

hostToolchain :: Ctx -> Toolchain
hostToolchain ctx =
  Toolchain
    { tcCC = depPath ctx "gcc" "bin/gcc"
    , tcCXX = depPath ctx "gcc" "bin/g++"
    , tcAR = depPath ctx "binutils" "bin/ar"
    , tcLD = depPath ctx "binutils" "bin/ld"
    , tcStrip = depPath ctx "binutils" "bin/strip"
    , tcTriple = Nothing
    }

crossToolchain :: Ctx -> Triple -> Toolchain
crossToolchain ctx target =
  let prefix = toString target <> "-"
      toolchainPkg = "gcc-cross-" <> toString target
   in Toolchain
        { tcCC = depPath ctx toolchainPkg $ "bin/" <> prefix <> "gcc"
        , tcCXX = depPath ctx toolchainPkg $ "bin/" <> prefix <> "g++"
        , tcAR = depPath ctx toolchainPkg $ "bin/" <> prefix <> "ar"
        , tcLD = depPath ctx toolchainPkg $ "bin/" <> prefix <> "ld"
        , tcStrip = depPath ctx toolchainPkg $ "bin/" <> prefix <> "strip"
        , tcTriple = Just target
        }

-- | CMake flags for cross compilation
crossCMakeFlags :: Toolchain -> [String]
crossCMakeFlags tc = case tcTriple tc of
  Nothing -> []
  Just target ->
    [ "-DCMAKE_SYSTEM_NAME=" <> osName (os target)
    , "-DCMAKE_SYSTEM_PROCESSOR=" <> archName (arch target)
    , "-DCMAKE_C_COMPILER=" <> tcCC tc
    , "-DCMAKE_CXX_COMPILER=" <> tcCXX tc
    , "-DCMAKE_AR=" <> tcAR tc
    ]
  where
    osName = \case
      Linux -> "Linux"
      Darwin -> "Darwin"
      Windows -> "Windows"
      _ -> "Generic"
    archName = \case
      X86_64 -> "x86_64"
      AArch64 -> "aarch64"
      ARMv7 -> "arm"
      RISCV64 -> "riscv64"
      WASM32 -> "wasm32"
      PowerPC64LE -> "ppc64le"

-- | Autotools flags for cross compilation
crossAutotoolsFlags :: Toolchain -> [String]
crossAutotoolsFlags tc = case tcTriple tc of
  Nothing -> []
  Just target ->
    [ "--host=" <> toString target
    , "CC=" <> tcCC tc
    , "CXX=" <> tcCXX tc
    , "AR=" <> tcAR tc
    , "LD=" <> tcLD tc
    ]

--------------------------------------------------------------------------------
-- Checks
--------------------------------------------------------------------------------

-- | Run a check
check :: String -> IO Bool -> IO ()
check name test = do
  log' $ "Check: " <> name
  ok <- test
  unless ok $ failBuild $ "Check failed: " <> name

-- | Fail the build
failBuild :: String -> IO a
failBuild msg = do
  hPutStrLn stderr $ "[FAIL] " <> msg
  exitFailure

-- | Check no shared libraries in output
noSharedLibs :: Ctx -> IO ()
noSharedLibs ctx = check "no-shared-libs" $ do
  libs <- glob (outPath ctx "lib/*")
  let shared = filter isShared libs
  when (not $ null shared) $
    hPutStrLn stderr $
      "Found shared libraries: " <> unwords shared
  pure $ null shared
  where
    isShared f = takeExtension f `elem` [".so", ".dylib", ".dll"]

-- | Check file exists
hasFile :: Ctx -> FilePath -> IO ()
hasFile ctx f = check ("has-file: " <> f) $ doesFileExist (outPath ctx f)

-- | Check directory exists
hasDir :: Ctx -> FilePath -> IO ()
hasDir ctx d = check ("has-dir: " <> d) $ doesDirectoryExist (outPath ctx d)

--------------------------------------------------------------------------------
-- Logging
--------------------------------------------------------------------------------

log' :: String -> IO ()
log' msg = hPutStrLn stderr $ "[aleph] " <> msg
