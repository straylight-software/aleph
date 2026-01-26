{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TemplateHaskell #-}

{- |
Module      : Armitage.Toolchain
Description : Extract toolchain from Nix stdenv

Given a Nix stdenv (or any derivation with a compiler), extract
the toolchain definition that Buck2 needs:

  - Compiler paths (cc, cxx, ar, ld)
  - Include paths (clang resource dir, libstdc++, glibc)
  - Library paths (for linking)
  - Flags (from NIX_CFLAGS_COMPILE, etc.)

Output is line-oriented for easy consumption:

  CC=/nix/store/.../bin/clang
  CXX=/nix/store/.../bin/clang++
  AR=/nix/store/.../bin/llvm-ar
  LD=/nix/store/.../bin/ld.lld
  CFLAGS=-isystem/nix/store/.../include ...
  LDFLAGS=-L/nix/store/.../lib -Wl,-rpath,...

This lets Buck2 actions use the toolchain without reimplementing
the stdenv logic in Starlark.
-}
module Armitage.Toolchain
  ( -- * Toolchain extraction
    Toolchain (..)
  , extractToolchain
  , toolchainToEnv
  , toolchainToFlags
  
    -- * Stdenv analysis
  , StdenvInfo (..)
  , analyzeStdenv
  ) where

import Control.Exception (try, SomeException)
import Data.Aeson (ToJSON, FromJSON, eitherDecode)
import qualified Data.Aeson as Aeson
import Data.ByteString.Lazy (ByteString)
import qualified Data.ByteString.Lazy.Char8 as BL8
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Char (toLower, isUpper)
import Data.Maybe (fromMaybe, catMaybes, mapMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import GHC.Generics (Generic)
import System.Directory (doesFileExist, doesDirectoryExist)
import System.Exit (ExitCode (..))
import System.FilePath ((</>), takeDirectory)
import System.Process (readProcessWithExitCode)

-- -----------------------------------------------------------------------------
-- Toolchain definition
-- -----------------------------------------------------------------------------

data Toolchain = Toolchain
  { tcCC :: Text              -- ^ C compiler path
  , tcCXX :: Text             -- ^ C++ compiler path
  , tcAR :: Text              -- ^ Archiver path
  , tcLD :: Text              -- ^ Linker path
  , tcNM :: Maybe Text        -- ^ nm path
  , tcObjcopy :: Maybe Text   -- ^ objcopy path
  , tcStrip :: Maybe Text     -- ^ strip path
  , tcCFlags :: [Text]        -- ^ C compiler flags
  , tcCXXFlags :: [Text]      -- ^ C++ compiler flags (includes tcCFlags)
  , tcLDFlags :: [Text]       -- ^ Linker flags
  , tcIncludePaths :: [Text]  -- ^ -isystem paths
  , tcLibPaths :: [Text]      -- ^ -L paths
  , tcRPaths :: [Text]        -- ^ -Wl,-rpath paths
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON, FromJSON)

-- | Convert toolchain to environment variable format
toolchainToEnv :: Toolchain -> Text
toolchainToEnv Toolchain{..} = T.unlines $
  [ "CC=" <> tcCC
  , "CXX=" <> tcCXX
  , "AR=" <> tcAR
  , "LD=" <> tcLD
  ] ++
  maybe [] (\x -> ["NM=" <> x]) tcNM ++
  maybe [] (\x -> ["OBJCOPY=" <> x]) tcObjcopy ++
  maybe [] (\x -> ["STRIP=" <> x]) tcStrip ++
  [ "CFLAGS=" <> T.unwords tcCFlags
  , "CXXFLAGS=" <> T.unwords tcCXXFlags
  , "LDFLAGS=" <> T.unwords tcLDFlags
  ]

-- | Convert toolchain to compiler flags (for direct use)
toolchainToFlags :: Toolchain -> Text
toolchainToFlags Toolchain{..} = T.unlines $
  map ("-isystem" <>) tcIncludePaths ++
  map ("-L" <>) tcLibPaths ++
  map ("-Wl,-rpath," <>) tcRPaths ++
  tcCXXFlags ++
  tcLDFlags

-- -----------------------------------------------------------------------------
-- Stdenv analysis
-- -----------------------------------------------------------------------------

data StdenvInfo = StdenvInfo
  { siStorePath :: Text
  , siCC :: Text
  , siCXX :: Text
  , siBintools :: Text
  , siLibc :: Text
  , siLibcDev :: Text
  , siTargetPlatform :: Text
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON)

-- | Strip "si" prefix and handle casing for JSON field names
-- siStorePath -> storePath, siCC -> cc, siCXX -> cxx, siLibcDev -> libcDev
stripSiPrefix :: String -> String
stripSiPrefix ('s':'i':rest) = lowercaseLeading rest
stripSiPrefix s = s

-- | Lowercase leading uppercase chars: "CC" -> "cc", "CXX" -> "cxx", "StorePath" -> "storePath"
lowercaseLeading :: String -> String
lowercaseLeading [] = []
lowercaseLeading (c:cs)
  | isUpper c = case cs of
      -- If next char is also uppercase or end of string, lowercase current
      (c':_) | isUpper c' -> toLower c : lowercaseLeading cs
      []                  -> [toLower c]
      -- If next char is lowercase, only lowercase current (e.g., "StorePath" -> "storePath")
      _                   -> toLower c : cs
  | otherwise = c : cs

instance FromJSON StdenvInfo where
  parseJSON = Aeson.genericParseJSON Aeson.defaultOptions
    { Aeson.fieldLabelModifier = stripSiPrefix }

-- | Analyze a stdenv to extract component paths
analyzeStdenv :: Text -> IO (Either Text StdenvInfo)
analyzeStdenv flakeRef = do
  -- Get the stdenv derivation
  (exitCode, stdout, stderr) <- readProcessWithExitCode
    "nix"
    [ "eval"
    , T.unpack flakeRef
    , "--json"
    , "--apply", stdenvQuery
    ]
    ""
  case exitCode of
    ExitSuccess -> 
      case eitherDecode (BL8.pack stdout) of
        Right info -> pure $ Right info
        Left err -> pure $ Left $ "Failed to parse stdenv info: " <> T.pack err
    ExitFailure _ -> 
      pure $ Left $ "Failed to evaluate stdenv: " <> T.pack stderr

-- | Nix expression to extract stdenv components
stdenvQuery :: String
stdenvQuery = unlines
  [ "stdenv: {"
  , "  storePath = stdenv.outPath or \"\";"
  , "  cc = stdenv.cc.outPath or \"\";"
  , "  cxx = stdenv.cc.outPath or \"\";"  -- Usually same as cc for clang/gcc
  , "  bintools = stdenv.cc.bintools.outPath or \"\";"
  , "  libc = stdenv.cc.libc.outPath or \"\";"
  , "  libcDev = stdenv.cc.libc.dev.outPath or stdenv.cc.libc.outPath or \"\";"
  , "  targetPlatform = stdenv.targetPlatform.config or \"x86_64-unknown-linux-gnu\";"
  , "}"
  ]

-- -----------------------------------------------------------------------------
-- Toolchain extraction
-- -----------------------------------------------------------------------------

-- | Extract a complete toolchain from a flake ref
-- Can be:
--   - A stdenv: nixpkgs#stdenv
--   - A package with passthru.stdenv: nixpkgs#hello
--   - A specific compiler: nixpkgs#llvmPackages_18.clang
extractToolchain :: Text -> IO (Either Text Toolchain)
extractToolchain flakeRef = do
  -- First try to get stdenv info
  stdenvResult <- analyzeStdenv (flakeRef <> ".stdenv")
  
  case stdenvResult of
    Right info -> buildToolchainFromStdenv info
    Left _ -> do
      -- Try direct stdenv
      directResult <- analyzeStdenv flakeRef
      case directResult of
        Right info -> buildToolchainFromStdenv info
        Left err -> pure $ Left err

-- | Build toolchain from stdenv info by probing the store paths
buildToolchainFromStdenv :: StdenvInfo -> IO (Either Text Toolchain)
buildToolchainFromStdenv StdenvInfo{..} = do
  -- Find compiler binaries
  let ccPath = siCC <> "/bin"
  
  -- Probe for clang vs gcc
  hasClang <- doesFileExist (T.unpack ccPath </> "clang")
  hasGcc <- doesFileExist (T.unpack ccPath </> "gcc")
  
  let (cc, cxx) = if hasClang
        then (ccPath <> "/clang", ccPath <> "/clang++")
        else (ccPath <> "/gcc", ccPath <> "/g++")
  
  -- Find bintools
  let bintoolsPath = siBintools <> "/bin"
  
  -- Probe for llvm vs gnu bintools
  hasLlvmAr <- doesFileExist (T.unpack bintoolsPath </> "llvm-ar")
  
  let (ar, ld, nm, objcopy, strip) = if hasLlvmAr
        then ( bintoolsPath <> "/llvm-ar"
             , bintoolsPath <> "/ld.lld"
             , Just $ bintoolsPath <> "/llvm-nm"
             , Just $ bintoolsPath <> "/llvm-objcopy"
             , Just $ bintoolsPath <> "/llvm-strip"
             )
        else ( bintoolsPath <> "/ar"
             , bintoolsPath <> "/ld"
             , Just $ bintoolsPath <> "/nm"
             , Just $ bintoolsPath <> "/objcopy"
             , Just $ bintoolsPath <> "/strip"
             )
  
  -- Build include paths
  includePaths <- findIncludePaths siCC siLibcDev siTargetPlatform
  
  -- Build library paths
  libPaths <- findLibPaths siCC siLibc siTargetPlatform
  
  -- Build flags
  let cflags = map ("-isystem" <>) includePaths
      ldflags = 
        ["-fuse-ld=lld" | hasLlvmAr] ++
        concatMap (\p -> ["-L" <> p, "-Wl,-rpath," <> p]) libPaths
  
  pure $ Right Toolchain
    { tcCC = cc
    , tcCXX = cxx
    , tcAR = ar
    , tcLD = ld
    , tcNM = nm
    , tcObjcopy = objcopy
    , tcStrip = strip
    , tcCFlags = cflags
    , tcCXXFlags = cflags  -- C++ inherits C flags
    , tcLDFlags = ldflags
    , tcIncludePaths = includePaths
    , tcLibPaths = libPaths
    , tcRPaths = libPaths
    }

-- | Find include paths by probing the store
findIncludePaths :: Text -> Text -> Text -> IO [Text]
findIncludePaths ccPath libcDevPath targetPlatform = do
  paths <- sequence
    [ probeDir (ccPath <> "/lib/clang") >>= \case
        Just clangDir -> findClangResourceDir clangDir
        Nothing -> pure Nothing
    , probeDir (ccPath <> "/include/c++")
    , probeDir (ccPath <> "/include/c++/" <> extractVersion ccPath <> "/" <> targetPlatform)
    , probeDir (libcDevPath <> "/include")
    ]
  pure $ catMaybes paths

-- | Find library paths by probing the store
findLibPaths :: Text -> Text -> Text -> IO [Text]
findLibPaths ccPath libcPath targetPlatform = do
  paths <- sequence
    [ probeDir (ccPath <> "/lib")
    , probeDir (ccPath <> "/lib/gcc/" <> targetPlatform)
    , probeDir (libcPath <> "/lib")
    ]
  pure $ catMaybes paths

-- | Probe if a directory exists and return it
probeDir :: Text -> IO (Maybe Text)
probeDir path = do
  exists <- doesDirectoryExist (T.unpack path)
  pure $ if exists then Just path else Nothing

-- | Find clang resource directory (contains __stddef.h etc)
findClangResourceDir :: Text -> IO (Maybe Text)
findClangResourceDir clangLibDir = do
  -- clang resource dir is like /nix/store/.../lib/clang/18/include
  -- We need to find the version subdirectory
  (exitCode, stdout, _) <- readProcessWithExitCode
    "ls"
    [T.unpack clangLibDir]
    ""
  case exitCode of
    ExitSuccess -> do
      let versions = filter (T.all (`elem` ("0123456789." :: String))) $ 
                     T.lines $ T.pack stdout
      case versions of
        (v:_) -> pure $ Just $ clangLibDir <> "/" <> v
        [] -> pure Nothing
    _ -> pure Nothing

-- | Extract version from a store path like /nix/store/...-gcc-15.2.0/...
extractVersion :: Text -> Text
extractVersion path = 
  case T.breakOn "-" (T.reverse $ T.takeWhile (/= '/') $ T.reverse path) of
    (_, rest) -> T.takeWhile (/= '-') $ T.drop 1 rest
