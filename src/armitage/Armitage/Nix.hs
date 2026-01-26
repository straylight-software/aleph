{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Armitage.Nix
Description : Nix flake analysis for Buck2 integration

Analysis modes for Buck2 rules:

1. @resolve@ - Resolve a flake ref to compiler flags (preferred)
   Outputs flags directly to stdout, one per line:
     -isystem /nix/store/.../include
     -L/nix/store/.../lib
     -lz
   
   No JSON, no parsing. Buck2 just captures stdout.

2. @unroll@ - Unroll a nix derivation into concrete build commands.
   TODO[b7r6]: JSON output, replace with typed format later

3. @deps@ - Get the dependency graph for a flake ref.
   TODO[b7r6]: JSON output, replace with typed format later

Usage from Buck2:
  flags=$($(exe //src/armitage:nix-analyze) resolve nixpkgs#zlib)
  clang++ $flags main.cpp -o out

The key insight: Buck2 builds the analysis graph, we just provide
the information. No DICE in Haskell, Buck2's DICE handles incrementality.
-}
module Armitage.Nix
  ( -- * Analysis Modes
    AnalysisMode (..)
  , runAnalysis

    -- * Resolve Output (preferred - no JSON)
  , resolveToFlags

    -- * Unroll Output (TODO: replace JSON)
  , UnrollResult (..)
  , BuildSystem (..)
  , ResolvedPaths (..)

    -- * Deps Output (TODO: replace JSON)
  , DepsResult (..)
  , NixDep (..)
  , DepType (..)

    -- * Flake Resolution
  , resolveFlakeRef
  , getDerivationInfo
  , getBuildInputs
  ) where

import Control.Exception (try, SomeException)
import Control.Monad (forM)
import Data.Aeson (ToJSON (..), FromJSON (..), (.=), object, encode, eitherDecode)
import qualified Data.Aeson as Aeson
import Data.ByteString.Lazy (ByteString)
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Lazy.Char8 as BL8
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe, catMaybes)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.IO as TIO
import GHC.Generics (Generic)
import System.Exit (ExitCode (..))
import System.Process (readProcessWithExitCode)

-- -----------------------------------------------------------------------------
-- Analysis Modes
-- -----------------------------------------------------------------------------

data AnalysisMode
  = ModeResolve [Text]  -- ^ Resolve flake refs to compiler flags
  | ModeUnroll Text     -- ^ Unroll build system for flake ref
  | ModeDeps Text       -- ^ Get dependencies for flake ref
  deriving (Show, Eq)

-- | Run analysis and return result
-- For ModeResolve, returns flags as newline-separated text (not JSON)
-- For others, returns JSON (TODO: replace)
runAnalysis :: AnalysisMode -> IO ByteString
runAnalysis = \case
  ModeResolve refs -> BL.fromStrict . TE.encodeUtf8 <$> resolveToFlags refs
  ModeUnroll ref -> encode <$> unrollFlakeRef ref
  ModeDeps ref -> encode <$> getDepsForRef ref

-- -----------------------------------------------------------------------------
-- Resolve: Output compiler flags directly (preferred mode)
-- -----------------------------------------------------------------------------

-- | Resolve multiple flake refs to compiler/linker flags
-- Output format (one flag per line):
--   -isystem /nix/store/xxx/include
--   -L/nix/store/xxx/lib
--   -lz
--   -isystem /nix/store/yyy/include
--   -L/nix/store/yyy/lib
--   -lssl
--   -lcrypto
resolveToFlags :: [Text] -> IO Text
resolveToFlags refs = do
  flagSets <- mapM resolveOnePackage refs
  pure $ T.unlines (concat flagSets)

-- | Resolve a single package to its flags
resolveOnePackage :: Text -> IO [Text]
resolveOnePackage ref = do
  -- Get output path (try .out first for packages with multiple outputs)
  outPath <- resolveOutput ref "out"
  devPath <- resolveOutput ref "dev"
  
  let includePath = devPath <> "/include"
      libPath = outPath <> "/lib"
      libNames = getLibNames ref
  
  -- Build flag list
  pure $ 
    [ "-isystem", includePath
    , "-L" <> libPath
    ] ++ map ("-l" <>) libNames

-- | Resolve a specific output of a flake ref
resolveOutput :: Text -> Text -> IO Text
resolveOutput ref output = do
  -- Try ref.output first, fall back to ref
  let refWithOutput = ref <> "." <> output
  (exitCode, stdout, _) <- readProcessWithExitCode
    "nix"
    ["build", T.unpack refWithOutput, "--print-out-paths", "--no-link"]
    ""
  case exitCode of
    ExitSuccess -> pure $ T.strip $ T.pack stdout
    ExitFailure _ -> do
      -- Fall back to base ref
      (exitCode', stdout', _) <- readProcessWithExitCode
        "nix"
        ["build", T.unpack ref, "--print-out-paths", "--no-link"]
        ""
      case exitCode' of
        ExitSuccess -> pure $ T.strip $ T.pack stdout'
        ExitFailure _ -> error $ "Failed to resolve " <> T.unpack ref

-- | Get library names for a package
-- TODO[b7r6]: This should come from nix metadata, not a hardcoded map
getLibNames :: Text -> [Text]
getLibNames ref = 
  let pkg = T.takeWhileEnd (/= '#') ref
      -- Strip any output suffix (.dev, .out, etc)
      basePkg = T.takeWhile (/= '.') pkg
  in case Map.lookup basePkg libNameMap of
       Just libs -> libs
       Nothing -> [basePkg]  -- Assume lib name matches package name

-- | Map package names to library names
-- Some packages have different lib names than package names
-- Some packages provide multiple libraries
libNameMap :: Map Text [Text]
libNameMap = Map.fromList
  [ ("zlib", ["z"])
  , ("openssl", ["ssl", "crypto"])
  , ("libpng", ["png"])
  , ("libjpeg", ["jpeg"])
  , ("sqlite", ["sqlite3"])
  , ("curl", ["curl"])
  , ("boost", ["boost_system"])
  , ("protobuf", ["protobuf"])
  , ("lz4", ["lz4"])
  , ("zstd", ["zstd"])
  , ("brotli", ["brotlienc", "brotlidec", "brotlicommon"])
  , ("libsodium", ["sodium"])
  , ("gmp", ["gmp"])
  , ("mpfr", ["mpfr"])
  , ("pcre2", ["pcre2-8"])
  , ("libffi", ["ffi"])
  , ("libuv", ["uv"])
  , ("libgit2", ["git2"])
  , ("jq", ["jq"])
  , ("simdjson", ["simdjson"])
  , ("fmt", ["fmt"])
  , ("spdlog", ["spdlog"])
  , ("catch2", ["Catch2Main", "Catch2"])
  , ("nlohmann_json", [])  -- header-only
  , ("rapidjson", [])      -- header-only
  ]

-- -----------------------------------------------------------------------------
-- Unroll: Convert nix derivation to concrete build commands
-- TODO[b7r6]: Replace JSON with typed output
-- -----------------------------------------------------------------------------

data UnrollResult = UnrollResult
  { urFlakeRef :: Text
  , urStorePath :: Text
  , urBuildSystem :: BuildSystem
  , urPaths :: ResolvedPaths
  , urBuildCommands :: [Text]
  , urOutputs :: Map Text Text
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON, FromJSON)

data BuildSystem
  = BSCMake
  | BSAutotools
  | BSMeson
  | BSCargo
  | BSSetupPy
  | BSMakefile
  | BSCustom Text
  | BSPrebuilt
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON, FromJSON)

data ResolvedPaths = ResolvedPaths
  { rpIncludeDirs :: [Text]
  , rpLibDirs :: [Text]
  , rpLibs :: [Text]
  , rpPkgConfigPath :: [Text]
  , rpBinDirs :: [Text]
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON, FromJSON)

unrollFlakeRef :: Text -> IO UnrollResult
unrollFlakeRef ref = do
  (outPath, outputs) <- resolveFlakeRef ref
  drvInfo <- getDerivationInfo ref
  let buildSystem = detectBuildSystem drvInfo
  paths <- resolvePaths ref drvInfo
  let commands = generateBuildCommands buildSystem drvInfo
  pure UnrollResult
    { urFlakeRef = ref
    , urStorePath = outPath
    , urBuildSystem = buildSystem
    , urPaths = paths
    , urBuildCommands = commands
    , urOutputs = outputs
    }

detectBuildSystem :: Map Text Aeson.Value -> BuildSystem
detectBuildSystem attrs =
  case Map.lookup "buildPhase" attrs of
    Just (Aeson.String bp)
      | "cmake" `T.isInfixOf` bp -> BSCMake
      | "autoreconf" `T.isInfixOf` bp -> BSAutotools
      | "configure" `T.isInfixOf` bp -> BSAutotools
      | "meson" `T.isInfixOf` bp -> BSMeson
      | "cargo" `T.isInfixOf` bp -> BSCargo
      | "setup.py" `T.isInfixOf` bp -> BSSetupPy
    _ -> case Map.lookup "builder" attrs of
      Just (Aeson.String b)
        | "cmake" `T.isInfixOf` b -> BSCMake
        | "meson" `T.isInfixOf` b -> BSMeson
      _ -> BSPrebuilt

generateBuildCommands :: BuildSystem -> Map Text Aeson.Value -> [Text]
generateBuildCommands bs attrs = case bs of
  BSCMake -> cmakeCommands attrs
  BSAutotools -> autotoolsCommands attrs
  BSMeson -> mesonCommands attrs
  BSCargo -> cargoCommands attrs
  BSPrebuilt -> []
  _ -> []

cmakeCommands :: Map Text Aeson.Value -> [Text]
cmakeCommands attrs =
  let srcDir = getTextAttr "src" attrs
      cmakeFlags = getTextListAttr "cmakeFlags" attrs
  in [ "cmake -B build -S " <> fromMaybe "." srcDir <> " " <> T.unwords cmakeFlags
     , "cmake --build build -j$(nproc)"
     , "cmake --install build --prefix $out"
     ]

autotoolsCommands :: Map Text Aeson.Value -> [Text]
autotoolsCommands attrs =
  let configureFlags = getTextListAttr "configureFlags" attrs
  in [ "./configure --prefix=$out " <> T.unwords configureFlags
     , "make -j$(nproc)"
     , "make install"
     ]

mesonCommands :: Map Text Aeson.Value -> [Text]
mesonCommands attrs =
  let mesonFlags = getTextListAttr "mesonFlags" attrs
  in [ "meson setup build " <> T.unwords mesonFlags
     , "meson compile -C build"
     , "meson install -C build"
     ]

cargoCommands :: Map Text Aeson.Value -> [Text]
cargoCommands _ =
  [ "cargo build --release"
  , "cargo install --path . --root $out"
  ]

resolvePaths :: Text -> Map Text Aeson.Value -> IO ResolvedPaths
resolvePaths ref attrs = do
  (_, outputs) <- resolveFlakeRef ref
  let outPath = Map.findWithDefault "" "out" outputs
      devPath = Map.findWithDefault outPath "dev" outputs
  propagated <- getPropagatedPaths ref
  pure ResolvedPaths
    { rpIncludeDirs = [devPath <> "/include"] <> propagated
    , rpLibDirs = [outPath <> "/lib"]
    , rpLibs = extractLibNames attrs
    , rpPkgConfigPath = [outPath <> "/lib/pkgconfig", devPath <> "/lib/pkgconfig"]
    , rpBinDirs = [outPath <> "/bin"]
    }

extractLibNames :: Map Text Aeson.Value -> [Text]
extractLibNames attrs =
  case Map.lookup "pname" attrs of
    Just (Aeson.String pname) -> 
      case Map.lookup pname libNameMap of
        Just libs -> libs
        Nothing -> [pname]
    _ -> []

getPropagatedPaths :: Text -> IO [Text]
getPropagatedPaths ref = do
  inputs <- getBuildInputs ref
  paths <- forM inputs $ \inp -> do
    result <- try @SomeException $ resolveOutput inp "dev"
    case result of
      Right devPath -> pure $ Just (devPath <> "/include")
      Left _ -> pure Nothing
  pure $ catMaybes paths

-- -----------------------------------------------------------------------------
-- Deps: Get dependency graph
-- TODO[b7r6]: Replace JSON with typed output
-- -----------------------------------------------------------------------------

data DepsResult = DepsResult
  { drFlakeRef :: Text
  , drDeps :: [NixDep]
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON, FromJSON)

data NixDep = NixDep
  { ndName :: Text
  , ndFlakeRef :: Text
  , ndStorePath :: Text
  , ndType :: DepType
  , ndOutputs :: [Text]
  }
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON, FromJSON)

data DepType
  = DepBuildInput
  | DepPropagated
  | DepRuntime
  deriving stock (Show, Eq, Generic)
  deriving anyclass (ToJSON, FromJSON)

getDepsForRef :: Text -> IO DepsResult
getDepsForRef ref = do
  deps <- getBuildInputs ref
  nixDeps <- forM deps $ \depRef -> do
    result <- try @SomeException $ resolveFlakeRef depRef
    case result of
      Right (path, outputs) -> pure $ Just NixDep
        { ndName = extractName depRef
        , ndFlakeRef = depRef
        , ndStorePath = path
        , ndType = DepBuildInput
        , ndOutputs = Map.keys outputs
        }
      Left _ -> pure Nothing
  pure DepsResult
    { drFlakeRef = ref
    , drDeps = catMaybes nixDeps
    }

extractName :: Text -> Text
extractName ref =
  let afterHash = T.takeWhileEnd (/= '#') ref
  in T.takeWhile (/= '.') afterHash

-- -----------------------------------------------------------------------------
-- Nix Commands
-- -----------------------------------------------------------------------------

resolveFlakeRef :: Text -> IO (Text, Map Text Text)
resolveFlakeRef ref = do
  (exitCode, stdout, stderr) <- readProcessWithExitCode
    "nix"
    ["build", T.unpack ref, "--print-out-paths", "--no-link"]
    ""
  case exitCode of
    ExitSuccess -> do
      let outPath = T.strip $ T.pack stdout
      outputs <- getOutputPaths ref
      pure (outPath, Map.insert "out" outPath outputs)
    ExitFailure _ -> error $ "Failed to resolve " <> T.unpack ref <> ": " <> stderr

getOutputPaths :: Text -> IO (Map Text Text)
getOutputPaths ref = do
  let tryOutput name = do
        result <- try @SomeException $ do
          (exitCode, stdout, _) <- readProcessWithExitCode
            "nix"
            ["build", T.unpack (ref <> "." <> name), "--print-out-paths", "--no-link"]
            ""
          case exitCode of
            ExitSuccess -> pure $ Just (name, T.strip $ T.pack stdout)
            _ -> pure Nothing
        case result of
          Right mp -> pure mp
          Left _ -> pure Nothing
  outputs <- mapM tryOutput ["dev", "lib", "bin", "doc", "man"]
  pure $ Map.fromList $ catMaybes outputs

getDerivationInfo :: Text -> IO (Map Text Aeson.Value)
getDerivationInfo ref = do
  (exitCode, stdout, _) <- readProcessWithExitCode
    "nix"
    [ "eval"
    , T.unpack ref <> ".drvAttrs"
    , "--json"
    ]
    ""
  case exitCode of
    ExitSuccess ->
      case eitherDecode (BL8.pack stdout) of
        Right attrs -> pure attrs
        Left _ -> pure Map.empty
    ExitFailure _ -> pure Map.empty

getBuildInputs :: Text -> IO [Text]
getBuildInputs ref = do
  (exitCode, stdout, _) <- readProcessWithExitCode
    "nix"
    [ "eval"
    , T.unpack ref <> ".buildInputs"
    , "--json"
    , "--apply", "map (x: x.pname or x.name or \"unknown\")"
    ]
    ""
  case exitCode of
    ExitSuccess ->
      case eitherDecode @[Text] (BL8.pack stdout) of
        Right inputs -> pure $ map (\n -> "nixpkgs#" <> n) inputs
        Left _ -> pure []
    ExitFailure _ -> pure []

-- -----------------------------------------------------------------------------
-- Helpers
-- -----------------------------------------------------------------------------

getTextAttr :: Text -> Map Text Aeson.Value -> Maybe Text
getTextAttr key attrs = case Map.lookup key attrs of
  Just (Aeson.String t) -> Just t
  _ -> Nothing

getTextListAttr :: Text -> Map Text Aeson.Value -> [Text]
getTextListAttr key attrs = case Map.lookup key attrs of
  Just (Aeson.Array arr) -> catMaybes $ map extractText $ toList arr
  _ -> []
  where
    extractText (Aeson.String t) = Just t
    extractText _ = Nothing
    toList = foldr (:) []
