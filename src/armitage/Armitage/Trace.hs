{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Armitage.Trace
Description : Build system confession via strace interception

Ratfuck CMake: intercept compiler/linker calls, extract dependency graph,
emit typed Dhall targets. The build system confesses its sins.

Pipeline:
  1. strace -f -e execve <build command>
  2. Parse execve calls for compilers/linkers
  3. Extract -I, -L, -l, source files
  4. Resolve paths to flake refs where possible
  5. Emit Dhall targets

The key insight (proven in Continuity.lean §17):
  Build systems are parametric over artifacts.
  They can't inspect .o contents, only route them.
  Therefore: traced graph = real graph.
-}
module Armitage.Trace
  ( -- * Tracing
    traceCommand
  , TraceConfig (..)
  , defaultTraceConfig
  
    -- * Parsing
  , CompilerCall (..)
  , LinkerCall (..)
  , parseStrace
  , parseExecve
  
    -- * Analysis
  , BuildGraph (..)
  , Target (..)
  , Dep (..)
  , analyzeTrace
  
    -- * Codegen
  , toDhall
  , toDhallFile
  ) where

import Control.Monad (forM, when)
import Data.List (isPrefixOf, isSuffixOf, nub, partition)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (catMaybes, mapMaybe)
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import GHC.Generics (Generic)
import System.Exit (ExitCode (..))
import System.FilePath (takeBaseName, takeExtension, takeFileName, (</>))
import System.Process (readProcessWithExitCode)
import Text.Read (readMaybe)

-- -----------------------------------------------------------------------------
-- Configuration
-- -----------------------------------------------------------------------------

data TraceConfig = TraceConfig
  { tcCompilers :: [Text]      -- ^ Compiler executables to match
  , tcLinkers :: [Text]        -- ^ Linker executables to match
  , tcArchivers :: [Text]      -- ^ Archiver executables to match
  , tcNixStore :: Text         -- ^ Nix store prefix
  , tcVerbose :: Bool          -- ^ Print debug info
  }
  deriving stock (Show, Eq, Generic)

defaultTraceConfig :: TraceConfig
defaultTraceConfig = TraceConfig
  { tcCompilers = ["clang", "clang++", "gcc", "g++", "cc", "c++", "nvcc"]
  , tcLinkers = ["ld", "ld.lld", "ld.gold", "ld.bfd", "collect2", "clang", "clang++", "gcc", "g++"]
  , tcArchivers = ["ar", "llvm-ar", "ranlib"]
  , tcNixStore = "/nix/store"
  , tcVerbose = False
  }

-- -----------------------------------------------------------------------------
-- Traced Calls
-- -----------------------------------------------------------------------------

-- | A compiler invocation
data CompilerCall = CompilerCall
  { ccCompiler :: Text           -- ^ Full path to compiler
  , ccSources :: [Text]          -- ^ Source files compiled
  , ccOutput :: Maybe Text       -- ^ Output file (-o)
  , ccIncludes :: [Text]         -- ^ Include paths (-I)
  , ccDefines :: [Text]          -- ^ Defines (-D)
  , ccFlags :: [Text]            -- ^ Other flags
  , ccPid :: Int                 -- ^ Process ID
  }
  deriving stock (Show, Eq, Generic)

-- | A linker invocation
data LinkerCall = LinkerCall
  { lcLinker :: Text             -- ^ Full path to linker
  , lcObjects :: [Text]          -- ^ Object files
  , lcOutput :: Maybe Text       -- ^ Output file (-o)
  , lcLibPaths :: [Text]         -- ^ Library search paths (-L)
  , lcLibs :: [Text]             -- ^ Libraries (-l)
  , lcRpaths :: [Text]           -- ^ Runtime paths (-rpath)
  , lcFlags :: [Text]            -- ^ Other flags
  , lcPid :: Int                 -- ^ Process ID
  }
  deriving stock (Show, Eq, Generic)

-- -----------------------------------------------------------------------------
-- Tracing
-- -----------------------------------------------------------------------------

-- | Run a command under strace and capture execve calls
traceCommand :: TraceConfig -> [String] -> IO (Either Text Text)
traceCommand cfg cmd = do
  when (tcVerbose cfg) $
    putStrLn $ "Tracing: " <> unwords cmd
  
  -- strace -f: follow forks
  -- strace -e execve: only trace exec calls
  -- strace -s 10000: don't truncate strings
  -- strace -o: output to stderr (we capture it)
  (exitCode, _stdout, stderr) <- readProcessWithExitCode
    "strace"
    (["-f", "-e", "execve", "-s", "10000", "-o", "/dev/stderr", "--"] ++ cmd)
    ""
  
  case exitCode of
    ExitSuccess -> pure $ Right $ T.pack stderr
    ExitFailure code -> pure $ Right $ T.pack stderr  -- Build may fail, we still want the trace

-- -----------------------------------------------------------------------------
-- Parsing
-- -----------------------------------------------------------------------------

-- | Parse strace output into compiler/linker calls
parseStrace :: TraceConfig -> Text -> ([CompilerCall], [LinkerCall])
parseStrace cfg trace = 
  let lines_ = T.lines trace
      parsed = mapMaybe (parseExecve cfg) lines_
      (compiles, links) = partitionCalls cfg parsed
  in (compiles, links)

-- | Parse a single strace execve line
-- Format: PID execve("/path/to/exe", ["exe", "arg1", "arg2", ...], ...) = 0
parseExecve :: TraceConfig -> Text -> Maybe (Int, Text, [Text])
parseExecve _cfg line = do
  -- Extract PID
  let (pidPart, rest) = T.breakOn " " line
  pid <- readMaybe (T.unpack pidPart)
  
  -- Check for execve
  let rest' = T.stripStart rest
  if not ("execve(" `T.isPrefixOf` rest')
    then Nothing
    else do
      -- Extract path
      let afterExecve = T.drop 7 rest'  -- drop "execve("
      (path, afterPath) <- extractQuoted afterExecve
      
      -- Extract argv
      let afterComma = T.dropWhile (/= '[') afterPath
      args <- extractArgv afterComma
      
      -- Check for success (= 0)
      if "= 0" `T.isInfixOf` line || "= -1" `T.isInfixOf` line  -- -1 is ENOENT, still useful
        then Just (pid, path, args)
        else Nothing

-- | Extract a quoted string
extractQuoted :: Text -> Maybe (Text, Text)
extractQuoted t = do
  let t' = T.dropWhile (/= '"') t
  if T.null t' then Nothing
  else do
    let inner = T.drop 1 t'  -- drop opening quote
        (content, rest) = T.breakOn "\"" inner
    Just (content, T.drop 1 rest)

-- | Extract argv array
extractArgv :: Text -> Maybe [Text]
extractArgv t = do
  let t' = T.dropWhile (/= '[') t
  if T.null t' then Nothing
  else do
    let inner = T.drop 1 t'  -- drop [
        content = T.takeWhile (/= ']') inner
    Just $ parseArgvContent content

-- | Parse the content of an argv array
parseArgvContent :: Text -> [Text]
parseArgvContent content = go content []
  where
    go t acc
      | T.null t = reverse acc
      | otherwise =
          case extractQuoted t of
            Just (arg, rest) -> go (T.dropWhile (\c -> c == ',' || c == ' ') rest) (unescapeArg arg : acc)
            Nothing -> reverse acc
    
    unescapeArg = T.replace "\\\"" "\"" . T.replace "\\\\" "\\"

-- | Partition parsed calls into compilers and linkers
partitionCalls :: TraceConfig -> [(Int, Text, [Text])] -> ([CompilerCall], [LinkerCall])
partitionCalls cfg calls = 
  let compilerCalls = mapMaybe (toCompilerCall cfg) calls
      linkerCalls = mapMaybe (toLinkerCall cfg) calls
  in (compilerCalls, linkerCalls)

-- | Check if path matches a compiler
isCompiler :: TraceConfig -> Text -> Bool
isCompiler cfg path = 
  let exe = T.pack $ takeFileName $ T.unpack path
  in any (`T.isPrefixOf` exe) (tcCompilers cfg)

-- | Check if this looks like a compile (not link) invocation
isCompileInvocation :: [Text] -> Bool
isCompileInvocation args = "-c" `elem` args

-- | Check if path matches a linker
isLinker :: TraceConfig -> Text -> Bool
isLinker cfg path =
  let exe = T.pack $ takeFileName $ T.unpack path
  in any (`T.isPrefixOf` exe) (tcLinkers cfg)

-- | Convert to CompilerCall if it's a compile invocation
toCompilerCall :: TraceConfig -> (Int, Text, [Text]) -> Maybe CompilerCall
toCompilerCall cfg (pid, path, args)
  | isCompiler cfg path && isCompileInvocation args = Just CompilerCall
      { ccCompiler = path
      , ccSources = extractSources args
      , ccOutput = extractOutput args
      , ccIncludes = extractIncludes args
      , ccDefines = extractDefines args
      , ccFlags = extractFlags args
      , ccPid = pid
      }
  | otherwise = Nothing

-- | Convert to LinkerCall if it's a link invocation
toLinkerCall :: TraceConfig -> (Int, Text, [Text]) -> Maybe LinkerCall
toLinkerCall cfg (pid, path, args)
  | (isLinker cfg path || isCompiler cfg path) && not (isCompileInvocation args) && hasObjects args = Just LinkerCall
      { lcLinker = path
      , lcObjects = extractObjects args
      , lcOutput = extractOutput args
      , lcLibPaths = extractLibPaths args
      , lcLibs = extractLibs args
      , lcRpaths = extractRpaths args
      , lcFlags = extractFlags args
      , lcPid = pid
      }
  | otherwise = Nothing
  where
    hasObjects as = any (T.isSuffixOf ".o") as || any (T.isSuffixOf ".a") as

-- | Extract source files from args
extractSources :: [Text] -> [Text]
extractSources = filter isSource
  where
    isSource arg = any (`T.isSuffixOf` arg) [".c", ".cc", ".cpp", ".cxx", ".cu", ".m", ".mm"]
                   && not ("-" `T.isPrefixOf` arg)

-- | Extract -o argument
extractOutput :: [Text] -> Maybe Text
extractOutput [] = Nothing
extractOutput [_] = Nothing
extractOutput ("-o":out:_) = Just out
extractOutput (_:rest) = extractOutput rest

-- | Extract -I arguments
extractIncludes :: [Text] -> [Text]
extractIncludes [] = []
extractIncludes [_] = []
extractIncludes (arg:next:rest)
  | arg == "-I" = next : extractIncludes rest
  | "-I" `T.isPrefixOf` arg = T.drop 2 arg : extractIncludes (next:rest)
  | "-isystem" == arg = next : extractIncludes rest
  | otherwise = extractIncludes (next:rest)

-- | Extract -D arguments  
extractDefines :: [Text] -> [Text]
extractDefines [] = []
extractDefines [_] = []
extractDefines (arg:next:rest)
  | arg == "-D" = next : extractDefines rest
  | "-D" `T.isPrefixOf` arg = T.drop 2 arg : extractDefines (next:rest)
  | otherwise = extractDefines (next:rest)

-- | Extract -L arguments
extractLibPaths :: [Text] -> [Text]
extractLibPaths [] = []
extractLibPaths [_] = []
extractLibPaths (arg:next:rest)
  | arg == "-L" = next : extractLibPaths rest
  | "-L" `T.isPrefixOf` arg = T.drop 2 arg : extractLibPaths (next:rest)
  | otherwise = extractLibPaths (next:rest)

-- | Extract -l arguments
extractLibs :: [Text] -> [Text]
extractLibs [] = []
extractLibs [_] = []
extractLibs (arg:next:rest)
  | arg == "-l" = next : extractLibs rest
  | "-l" `T.isPrefixOf` arg = T.drop 2 arg : extractLibs (next:rest)
  | otherwise = extractLibs (next:rest)

-- | Extract -rpath arguments
extractRpaths :: [Text] -> [Text]
extractRpaths [] = []
extractRpaths [_] = []
extractRpaths (arg:next:rest)
  | arg == "-rpath" = next : extractRpaths rest
  | "-Wl,-rpath," `T.isPrefixOf` arg = T.drop 12 arg : extractRpaths (next:rest)
  | otherwise = extractRpaths (next:rest)

-- | Extract object files from args
extractObjects :: [Text] -> [Text]
extractObjects = filter isObject
  where
    isObject arg = (T.isSuffixOf ".o" arg || T.isSuffixOf ".a" arg)
                   && not ("-" `T.isPrefixOf` arg)

-- | Extract remaining flags
extractFlags :: [Text] -> [Text]
extractFlags = filter isFlag
  where
    isFlag arg = "-" `T.isPrefixOf` arg 
                 && not (any (`T.isPrefixOf` arg) ["-I", "-D", "-L", "-l", "-o", "-rpath", "-Wl,-rpath"])
                 && arg /= "-c"

-- -----------------------------------------------------------------------------
-- Analysis
-- -----------------------------------------------------------------------------

-- | A build target extracted from trace
data Target = Target
  { tName :: Text
  , tSources :: [Text]
  , tDeps :: [Dep]
  , tCompiler :: Text
  , tFlags :: [Text]
  }
  deriving stock (Show, Eq, Generic)

-- | A dependency
data Dep
  = DepFlake Text        -- ^ nixpkgs#foo
  | DepLocal Text        -- ^ Local path
  | DepSystem Text       -- ^ System library (-lfoo)
  deriving stock (Show, Eq, Generic)

-- | The complete build graph
data BuildGraph = BuildGraph
  { bgTargets :: [Target]
  , bgFlakeRefs :: Set Text   -- ^ All discovered flake refs
  }
  deriving stock (Show, Eq, Generic)

-- | Analyze trace into build graph
analyzeTrace :: TraceConfig -> ([CompilerCall], [LinkerCall]) -> IO BuildGraph
analyzeTrace cfg (compiles, links) = do
  -- Group compiles by output directory (heuristic for targets)
  -- Then match with link steps
  
  -- Deduplicate links by output path (clang driver invokes ld multiple times)
  let uniqueLinks = deduplicateLinks links
  
  -- For now, simple: each unique link = one target
  targets <- forM uniqueLinks $ \lc -> do
    -- Find compiles that produced the objects
    let objs = Set.fromList (lcObjects lc)
        relevantCompiles = filter (\cc -> maybe False (`Set.member` objs) (ccOutput cc)) compiles
        allSources = nub $ concatMap ccSources relevantCompiles
        allIncludes = nub $ concatMap ccIncludes relevantCompiles ++ lcLibPaths lc
    
    -- Convert includes/libpaths to deps
    deps <- mapM (pathToDep cfg) allIncludes
    let libDeps = map DepSystem (lcLibs lc)
    
    -- Target name from output
    let name = maybe "unknown" (T.pack . takeBaseName . T.unpack) (lcOutput lc)
    
    pure Target
      { tName = name
      , tSources = allSources
      , tDeps = nub $ catMaybes deps ++ libDeps
      , tCompiler = lcLinker lc
      , tFlags = nub $ concatMap ccFlags relevantCompiles ++ lcFlags lc
      }
  
  -- Collect all flake refs
  let flakeRefs = Set.fromList [ ref | t <- targets, DepFlake ref <- tDeps t ]
  
  pure BuildGraph
    { bgTargets = targets
    , bgFlakeRefs = flakeRefs
    }

-- | Deduplicate links by output, merging info from all invocations
deduplicateLinks :: [LinkerCall] -> [LinkerCall]
deduplicateLinks links =
  let byOutput = Map.fromListWith mergeLinks 
                   [ (out, lc) | lc <- links, Just out <- [lcOutput lc] ]
  in Map.elems byOutput
  where
    mergeLinks lc1 lc2 = LinkerCall
      { lcLinker = lcLinker lc1
      , lcObjects = nub $ lcObjects lc1 ++ lcObjects lc2
      , lcOutput = lcOutput lc1
      , lcLibPaths = nub $ lcLibPaths lc1 ++ lcLibPaths lc2
      , lcLibs = nub $ lcLibs lc1 ++ lcLibs lc2
      , lcRpaths = nub $ lcRpaths lc1 ++ lcRpaths lc2
      , lcFlags = nub $ lcFlags lc1 ++ lcFlags lc2
      , lcPid = lcPid lc1
      }

-- | Convert a path to a dependency
pathToDep :: TraceConfig -> Text -> IO (Maybe Dep)
pathToDep cfg path
  | tcNixStore cfg `T.isPrefixOf` path = do
      -- It's a nix store path - try to resolve to flake ref
      ref <- storePathToFlakeRef path
      pure $ Just $ DepFlake ref
  | otherwise = 
      pure $ Just $ DepLocal path

-- | Convert nix store path to flake ref
-- /nix/store/xxx-foo-1.2.3 -> nixpkgs#foo (heuristic)
storePathToFlakeRef :: Text -> IO Text
storePathToFlakeRef path = do
  -- Extract package name from store path
  -- Format: /nix/store/HASH-NAME-VERSION
  let parts = T.splitOn "-" $ T.drop 44 path  -- 44 = "/nix/store/" + 32-char hash + "-"
  case parts of
    [] -> pure $ "unknown#" <> path
    (name:_) -> pure $ "nixpkgs#" <> name  -- Assume nixpkgs, could query nix path-info

-- -----------------------------------------------------------------------------
-- Dhall Code Generation  
-- -----------------------------------------------------------------------------

-- | Generate Dhall source for build graph
toDhall :: BuildGraph -> Text
toDhall bg = 
  let targets = bgTargets bg
      -- Give each target a unique name
      indexedTargets = zip [1..] targets
      namedTargets = [ (uniqueName i t, t) | (i, t) <- indexedTargets ]
  in T.unlines $
    [ "-- Generated by armitage trace"
    , "-- " <> T.pack (show (length targets)) <> " target(s) extracted"
    , ""
    , "let Build = ./Build.dhall"
    , "let Toolchain = ./Toolchain.dhall"
    , ""
    ] ++ map (uncurry targetToDhallLet) namedTargets
      ++ [ "in {"
         , T.intercalate ",\n" [ "  " <> name <> " = " <> name | (name, _) <- namedTargets ]
         , "}"
         ]
  where
    uniqueName :: Int -> Target -> Text
    uniqueName i t = 
      let base = sanitizeName (tName t)
      in if length (bgTargets bg) == 1 then base else base <> "_" <> T.pack (show i)

-- | Generate Dhall let-binding for a single target
targetToDhallLet :: Text -> Target -> Text
targetToDhallLet varName Target {..} = T.unlines
  [ "let " <> varName <> " = Build.cxx-binary"
  , "  { name = \"" <> tName <> "\""
  , "  , srcs = " <> listToDhall tSources
  , "  , deps = " <> depsToDhall tDeps
  , "  , toolchain = Toolchain.presets.clang-18-glibc-dynamic"
  , "  , requires = []"
  , "  }"
  , ""
  ]

-- | Sanitize name for Dhall identifier
sanitizeName :: Text -> Text
sanitizeName = T.replace "-" "_" . T.replace "." "_"

-- | Convert list to Dhall
listToDhall :: [Text] -> Text
listToDhall [] = "[] : List Text"
listToDhall xs = "[" <> T.intercalate ", " (map quote xs) <> "]"
  where
    quote t = "\"" <> t <> "\""

-- | Convert deps to Dhall
depsToDhall :: [Dep] -> Text
depsToDhall [] = "[] : List Build.Dep"
depsToDhall deps = 
  let dhallDeps = mapMaybe depToDhall deps
  in if null dhallDeps 
     then "[] : List Build.Dep"
     else "[" <> T.intercalate ", " dhallDeps <> "]"

-- | Convert single dep to Dhall
depToDhall :: Dep -> Maybe Text
depToDhall = \case
  DepFlake ref -> Just $ "Build.dep.flake \"" <> ref <> "\""
  DepLocal _ -> Nothing  -- Skip local for now
  DepSystem lib -> Just $ "Build.dep.system \"" <> lib <> "\""

-- | Write Dhall to file
toDhallFile :: FilePath -> BuildGraph -> IO ()
toDhallFile path bg = TIO.writeFile path (toDhall bg)
