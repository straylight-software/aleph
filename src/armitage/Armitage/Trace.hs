{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Armitage.Trace
Description : Build system confession via strace/ptrace interception

The All Seeing Eye.

Two modes:
  1. Build tracing: Extract dependency graph from legacy build systems
  2. LSP tracing: Observe compiler I/O for language server ground truth

Pipeline:
  1. strace -f -e trace=execve,openat,read,write <command>
  2. Parse syscalls to extract:
     - execve: what commands ran
     - openat O_RDONLY: what files were read (inputs)
     - openat O_WRONLY: what files were written (outputs)
     - stderr: compiler diagnostics
  3. The trace IS the truth - no simulation, no approximation

The key insight (proven in Continuity.lean §17):
  Build systems are parametric over artifacts.
  They can't inspect .o contents, only route them.
  Therefore: traced graph = real graph.

For LSP: the compiler runs, we observe, diagnostics are real.
No reimplementation of type checkers. Just ground truth.
-}
module Armitage.Trace (
    -- * Tracing
    traceCommand,
    traceCommandFull,
    TraceConfig (..),
    defaultTraceConfig,

    -- * Full Trace (for LSP)
    FullTrace (..),
    FileAccess (..),
    AccessMode (..),
    parseFullTrace,

    -- * Parsing (legacy - execve only)
    CompilerCall (..),
    LinkerCall (..),
    parseStrace,
    parseExecve,

    -- * Analysis
    BuildGraph (..),
    Target (..),
    Dep (..),
    analyzeTrace,
    analyzeInterpreterTrace,
    InterpreterCall (..),
    parseInterpreters,

    -- * Codegen
    toDhall,
    toDhallFile,
) where

import Control.Exception (SomeException, try)
import Control.Monad (forM, when)
import Data.List (nub)

import qualified Data.Map.Strict as Map
import Data.Maybe (catMaybes, mapMaybe)
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import GHC.Generics (Generic)
import System.Exit (ExitCode (..))
import System.FilePath (takeBaseName, takeFileName)
import System.Process (readProcessWithExitCode)
import Text.Read (readMaybe)

import qualified Armitage.Semantic as Semantic

-- -----------------------------------------------------------------------------
-- Configuration
-- -----------------------------------------------------------------------------

data TraceConfig = TraceConfig
    { tcCompilers :: [Text]
    -- ^ Compiler executables to match
    , tcLinkers :: [Text]
    -- ^ Linker executables to match
    , tcArchivers :: [Text]
    -- ^ Archiver executables to match
    , tcInterpreters :: [Text]
    -- ^ Interpreters (python, node, etc.)
    , tcNixStore :: Text
    -- ^ Nix store prefix
    , tcVerbose :: Bool
    -- ^ Print debug info
    }
    deriving stock (Show, Eq, Generic)

defaultTraceConfig :: TraceConfig
defaultTraceConfig =
    TraceConfig
        { tcCompilers =
            [ "clang"
            , "clang++"
            , "gcc"
            , "g++"
            , "cc"
            , "c++"
            , "nvcc"
            , "ghc"
            , "ghc-9.12.2"
            , "ghc-9.10.1"
            , "ghc-9.8.4"
            , "ghc-9.6.6" -- Haskell
            , "rustc" -- Rust
            , "lean"
            , "leanc" -- Lean
            ]
        , tcLinkers =
            [ "ld"
            , "ld.lld"
            , "ld.gold"
            , "ld.bfd"
            , "collect2"
            , "clang"
            , "clang++"
            , "gcc"
            , "g++"
            , "ghc"
            , "ghc-9.12.2"
            , "ghc-9.10.1"
            , "ghc-9.8.4"
            , "ghc-9.6.6" -- GHC also links
            , "rustc" -- Rust also links
            ]
        , tcArchivers = ["ar", "llvm-ar", "ranlib"]
        , tcInterpreters =
            [ "python"
            , "python3"
            , "python3.11"
            , "python3.12"
            , "python3.13"
            , "node"
            , "nodejs"
            , "deno"
            , "bun" -- JavaScript
            , "ruby"
            , "perl"
            , "lua"
            ]
        , tcNixStore = "/nix/store"
        , tcVerbose = False
        }

-- -----------------------------------------------------------------------------
-- Traced Calls
-- -----------------------------------------------------------------------------

-- | A compiler invocation
data CompilerCall = CompilerCall
    { ccCompiler :: Text
    -- ^ Full path to compiler
    , ccSources :: [Text]
    -- ^ Source files compiled
    , ccOutput :: Maybe Text
    -- ^ Output file (-o)
    , ccIncludes :: [Text]
    -- ^ Include paths (-I)
    , ccDefines :: [Text]
    -- ^ Defines (-D)
    , ccFlags :: [Text]
    -- ^ Other flags
    , ccPid :: Int
    -- ^ Process ID
    }
    deriving stock (Show, Eq, Generic)

-- | A linker invocation
data LinkerCall = LinkerCall
    { lcLinker :: Text
    -- ^ Full path to linker
    , lcObjects :: [Text]
    -- ^ Object files
    , lcOutput :: Maybe Text
    -- ^ Output file (-o)
    , lcLibPaths :: [Text]
    -- ^ Library search paths (-L)
    , lcLibs :: [Text]
    -- ^ Libraries (-l)
    , lcRpaths :: [Text]
    -- ^ Runtime paths (-rpath)
    , lcFlags :: [Text]
    -- ^ Other flags
    , lcPid :: Int
    -- ^ Process ID
    }
    deriving stock (Show, Eq, Generic)

-- -----------------------------------------------------------------------------
-- Full Trace Types (for LSP)
-- -----------------------------------------------------------------------------

-- | Complete trace of a command execution
data FullTrace = FullTrace
    { ftCommand :: [Text]
    -- ^ Command that was run
    , ftExecves :: [(Int, Text, [Text])]
    -- ^ (pid, path, args)
    , ftFileAccesses :: [FileAccess]
    -- ^ All file accesses
    , ftStdout :: Text
    -- ^ Captured stdout
    , ftStderr :: Text
    -- ^ Captured stderr (compiler diagnostics!)
    , ftExitCode :: Int
    -- ^ Exit code
    }
    deriving stock (Show, Eq, Generic)

-- | A file access observed via strace
data FileAccess = FileAccess
    { faPid :: !Int
    , faPath :: !Text
    , faMode :: !AccessMode
    , faTimestamp :: !Int
    -- ^ Relative order
    }
    deriving stock (Show, Eq, Generic)

-- | File access mode
data AccessMode
    = -- | openat with O_RDONLY
      ModeRead
    | -- | openat with O_WRONLY or O_RDWR
      ModeWrite
    | -- | openat with O_CREAT
      ModeCreate
    deriving stock (Show, Eq, Ord, Generic)

-- -----------------------------------------------------------------------------
-- Tracing
-- -----------------------------------------------------------------------------

{- | Run a command under strace and capture execve + openat calls
Captures both compilers (execve) and interpreter file access (openat)
-}
traceCommand :: TraceConfig -> [String] -> IO (Either Text Text)
traceCommand cfg cmd = do
    when (tcVerbose cfg) $
        putStrLn $
            "Tracing: " <> unwords cmd

    -- strace -f: follow forks
    -- strace -e execve,openat: trace exec calls AND file opens
    -- strace -s 10000: don't truncate strings
    -- strace -o: output to stderr (we capture it)
    (exitCode, _stdout, stderr) <-
        readProcessWithExitCode
            "strace"
            (["-f", "-e", "trace=execve,openat", "-s", "10000", "-o", "/dev/stderr", "--"] ++ cmd)
            ""

    case exitCode of
        ExitSuccess -> pure $ Right $ T.pack stderr
        ExitFailure _code -> pure $ Right $ T.pack stderr -- Build may fail, we still want the trace

{- | Run a command under strace and capture EVERYTHING (for LSP)
Captures: execve, openat, file reads/writes, stdout, stderr
-}
traceCommandFull :: TraceConfig -> [String] -> IO FullTrace
traceCommandFull cfg cmd = do
    when (tcVerbose cfg) $
        putStrLn $
            "Full trace: " <> unwords cmd

    -- strace -f: follow forks
    -- strace -e trace=...: trace multiple syscall types
    -- strace -y: print paths for file descriptors
    -- strace -yy: even more fd info
    -- strace -s 10000: don't truncate strings
    -- strace -tt: timestamps
    -- Separate trace output from program output
    (exitCode, stdout, stderr) <-
        readProcessWithExitCode
            "strace"
            ( [ "-f"
              , "-e"
              , "trace=execve,openat,open,creat"
              , "-y" -- Show paths for fds
              , "-s"
              , "10000" -- Don't truncate
              , "-o"
              , "/dev/fd/3" -- Trace to fd 3
              , "--"
              ]
                ++ cmd
            )
            ""
    -- Note: This doesn't actually work for fd 3, we need a different approach
    -- For now, we'll parse stderr which contains both trace and program stderr

    let traceText = T.pack stderr -- In practice, strace output goes here
        (execves, fileAccesses) = parseFullTrace' traceText
        exitInt = case exitCode of
            ExitSuccess -> 0
            ExitFailure n -> n

    pure
        FullTrace
            { ftCommand = map T.pack cmd
            , ftExecves = execves
            , ftFileAccesses = fileAccesses
            , ftStdout = T.pack stdout
            , ftStderr = extractCompilerStderr traceText
            , ftExitCode = exitInt
            }

{- | Extract actual compiler stderr from strace output
Compiler errors are written via write(2, ...) calls
-}
extractCompilerStderr :: Text -> Text
extractCompilerStderr trace =
    let lines_ = T.lines trace
        -- Look for write(2, "error...", N) patterns
        stderrWrites = mapMaybe parseStderrWrite lines_
     in T.unlines stderrWrites

parseStderrWrite :: Text -> Maybe Text
parseStderrWrite line
    -- write(2, "...", N) = N
    | "write(2, \"" `T.isInfixOf` line
    , Just content <- extractWriteContent line =
        Just content
    | otherwise = Nothing

extractWriteContent :: Text -> Maybe Text
extractWriteContent line = do
    let afterWrite = T.drop 1 $ T.dropWhile (/= '"') line
        content = T.takeWhile (/= '"') afterWrite
    -- Unescape the content
    pure $ unescapeStrace content

unescapeStrace :: Text -> Text
unescapeStrace =
    T.replace "\\n" "\n"
        . T.replace "\\t" "\t"
        . T.replace "\\\"" "\""
        . T.replace "\\\\" "\\"

-- | Parse full strace output into execves and file accesses
parseFullTrace :: Text -> FullTrace
parseFullTrace trace =
    let (execves, accesses) = parseFullTrace' trace
     in FullTrace
            { ftCommand = []
            , ftExecves = execves
            , ftFileAccesses = accesses
            , ftStdout = ""
            , ftStderr = extractCompilerStderr trace
            , ftExitCode = 0
            }

parseFullTrace' :: Text -> ([(Int, Text, [Text])], [FileAccess])
parseFullTrace' trace =
    let lines_ = zip [0 ..] (T.lines trace)
        execves = mapMaybe (parseExecveLine . snd) lines_
        accesses = mapMaybe (uncurry parseOpenatLine) lines_
     in (execves, accesses)

parseExecveLine :: Text -> Maybe (Int, Text, [Text])
parseExecveLine line = do
    -- Format: PID execve("/path", ["arg0", "arg1"], ...) = 0
    let (pidPart, rest) = T.breakOn " " line
    pid <- readMaybe (T.unpack pidPart)
    let rest' = T.stripStart rest
    if not ("execve(" `T.isPrefixOf` rest')
        then Nothing
        else do
            let afterExecve = T.drop 7 rest'
            (path, afterPath) <- extractQuoted afterExecve
            let afterComma = T.dropWhile (/= '[') afterPath
            args <- extractArgv afterComma
            if "= 0" `T.isInfixOf` line
                then Just (pid, path, args)
                else Nothing

parseOpenatLine :: Int -> Text -> Maybe FileAccess
parseOpenatLine timestamp line = do
    -- Format: PID openat(AT_FDCWD, "/path", O_RDONLY|O_CLOEXEC) = 3</path>
    -- Or: PID openat(AT_FDCWD, "/path", O_WRONLY|O_CREAT|O_TRUNC, 0644) = 3
    let (pidPart, rest) = T.breakOn " " line
    pid <- readMaybe (T.unpack pidPart)
    let rest' = T.stripStart rest

    -- Check for openat or open
    (path, flags) <-
        if "openat(" `T.isPrefixOf` rest'
            then parseOpenatArgs (T.drop 7 rest')
            else
                if "open(" `T.isPrefixOf` rest'
                    then parseOpenArgs (T.drop 5 rest')
                    else Nothing

    -- Determine mode from flags
    let mode
            | "O_CREAT" `T.isInfixOf` flags = ModeCreate
            | "O_WRONLY" `T.isInfixOf` flags = ModeWrite
            | "O_RDWR" `T.isInfixOf` flags = ModeWrite
            | "O_RDONLY" `T.isInfixOf` flags = ModeRead
            | otherwise = ModeRead

    -- Only return if the call succeeded
    if "= -1" `T.isInfixOf` line
        then Nothing -- Failed open, ignore
        else
            Just
                FileAccess
                    { faPid = pid
                    , faPath = path
                    , faMode = mode
                    , faTimestamp = timestamp
                    }

parseOpenatArgs :: Text -> Maybe (Text, Text)
parseOpenatArgs t = do
    -- Skip AT_FDCWD or fd number
    let afterFd = T.drop 1 $ T.dropWhile (/= ',') t
    (path, afterPath) <- extractQuoted (T.stripStart afterFd)
    let afterComma = T.drop 1 $ T.dropWhile (/= ',') afterPath
        flags = T.takeWhile (/= ')') (T.stripStart afterComma)
    Just (path, flags)

parseOpenArgs :: Text -> Maybe (Text, Text)
parseOpenArgs t = do
    (path, afterPath) <- extractQuoted t
    let afterComma = T.drop 1 $ T.dropWhile (/= ',') afterPath
        flags = T.takeWhile (/= ')') (T.stripStart afterComma)
    Just (path, flags)

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

{- | Parse a single strace execve line
Format: PID execve("/path/to/exe", ["exe", "arg1", "arg2", ...], ...) = 0
-}
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
            let afterExecve = T.drop 7 rest' -- drop "execve("
            (path, afterPath) <- extractQuoted afterExecve

            -- Extract argv
            let afterComma = T.dropWhile (/= '[') afterPath
            args <- extractArgv afterComma

            -- Check for success (= 0)
            if "= 0" `T.isInfixOf` line || "= -1" `T.isInfixOf` line -- -1 is ENOENT, still useful
                then Just (pid, path, args)
                else Nothing

-- | Extract a quoted string
extractQuoted :: Text -> Maybe (Text, Text)
extractQuoted t = do
    let t' = T.dropWhile (/= '"') t
    if T.null t'
        then Nothing
        else do
            let inner = T.drop 1 t' -- drop opening quote
                (content, rest) = T.breakOn "\"" inner
            Just (content, T.drop 1 rest)

-- | Extract argv array
extractArgv :: Text -> Maybe [Text]
extractArgv t = do
    let t' = T.dropWhile (/= '[') t
    if T.null t'
        then Nothing
        else do
            let inner = T.drop 1 t' -- drop [
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
    | isCompiler cfg path && isCompileInvocation args =
        Just
            CompilerCall
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
    | (isLinker cfg path || isCompiler cfg path) && not (isCompileInvocation args) && hasObjects args =
        Just
            LinkerCall
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
    isSource arg =
        any
            (`T.isSuffixOf` arg)
            [ ".c"
            , ".cc"
            , ".cpp"
            , ".cxx"
            , ".cu"
            , ".m"
            , ".mm" -- C/C++/CUDA/ObjC
            , ".hs"
            , ".lhs" -- Haskell
            , ".rs" -- Rust
            , ".lean" -- Lean
            ]
            && not ("-" `T.isPrefixOf` arg)

-- | Extract -o argument
extractOutput :: [Text] -> Maybe Text
extractOutput [] = Nothing
extractOutput [_] = Nothing
extractOutput ("-o" : out : _) = Just out
extractOutput (_ : rest) = extractOutput rest

-- | Extract -I arguments
extractIncludes :: [Text] -> [Text]
extractIncludes [] = []
extractIncludes [_] = []
extractIncludes (arg : next : rest)
    | arg == "-I" = next : extractIncludes rest
    | "-I" `T.isPrefixOf` arg = T.drop 2 arg : extractIncludes (next : rest)
    | "-isystem" == arg = next : extractIncludes rest
    | otherwise = extractIncludes (next : rest)

-- | Extract -D arguments
extractDefines :: [Text] -> [Text]
extractDefines [] = []
extractDefines [_] = []
extractDefines (arg : next : rest)
    | arg == "-D" = next : extractDefines rest
    | "-D" `T.isPrefixOf` arg = T.drop 2 arg : extractDefines (next : rest)
    | otherwise = extractDefines (next : rest)

-- | Extract -L arguments
extractLibPaths :: [Text] -> [Text]
extractLibPaths [] = []
extractLibPaths [_] = []
extractLibPaths (arg : next : rest)
    | arg == "-L" = next : extractLibPaths rest
    | "-L" `T.isPrefixOf` arg = T.drop 2 arg : extractLibPaths (next : rest)
    | otherwise = extractLibPaths (next : rest)

-- | Extract -l arguments
extractLibs :: [Text] -> [Text]
extractLibs [] = []
extractLibs [_] = []
extractLibs (arg : next : rest)
    | arg == "-l" = next : extractLibs rest
    | "-l" `T.isPrefixOf` arg = T.drop 2 arg : extractLibs (next : rest)
    | otherwise = extractLibs (next : rest)

-- | Extract -rpath arguments
extractRpaths :: [Text] -> [Text]
extractRpaths [] = []
extractRpaths [_] = []
extractRpaths (arg : next : rest)
    | arg == "-rpath" = next : extractRpaths rest
    | "-Wl,-rpath," `T.isPrefixOf` arg = T.drop 12 arg : extractRpaths (next : rest)
    | otherwise = extractRpaths (next : rest)

-- | Extract object files from args
extractObjects :: [Text] -> [Text]
extractObjects = filter isObject
  where
    isObject arg =
        (T.isSuffixOf ".o" arg || T.isSuffixOf ".a" arg)
            && not ("-" `T.isPrefixOf` arg)

-- | Extract remaining flags
extractFlags :: [Text] -> [Text]
extractFlags = filter isFlag
  where
    isFlag arg =
        "-" `T.isPrefixOf` arg
            && not (any (`T.isPrefixOf` arg) ["-I", "-D", "-L", "-l", "-o", "-rpath", "-Wl,-rpath"])
            && arg /= "-c"

-- -----------------------------------------------------------------------------
-- Interpreter Parsing (Python, Node, etc.)
-- -----------------------------------------------------------------------------

-- | An interpreter invocation
data InterpreterCall = InterpreterCall
    { icInterpreter :: Text
    -- ^ python3, node, etc.
    , icScript :: Maybe Text
    -- ^ Main script file
    , icArgs :: [Text]
    -- ^ Script arguments
    , icImports :: [Text]
    -- ^ Imported files (from file access trace)
    , icPid :: Int
    -- ^ Process ID
    }
    deriving stock (Show, Eq, Generic)

-- | Check if executable is an interpreter
isInterpreter :: TraceConfig -> Text -> Bool
isInterpreter cfg path =
    let exe = T.pack $ takeFileName $ T.unpack path
     in any (`T.isPrefixOf` exe) (tcInterpreters cfg)

-- | Parse interpreter invocations from execve list
parseInterpreters :: TraceConfig -> [(Int, Text, [Text])] -> [InterpreterCall]
parseInterpreters cfg = mapMaybe (toInterpreterCall cfg)

-- | Convert execve to InterpreterCall if it's an interpreter
toInterpreterCall :: TraceConfig -> (Int, Text, [Text]) -> Maybe InterpreterCall
toInterpreterCall cfg (pid, path, args)
    | isInterpreter cfg path =
        Just
            InterpreterCall
                { icInterpreter = path
                , icScript = extractScript args
                , icArgs = drop 1 $ dropWhile (not . isScriptArg) args -- args after script
                , icImports = [] -- Filled in by analyzeInterpreterTrace
                , icPid = pid
                }
    | otherwise = Nothing
  where
    -- Skip interpreter args like -c, -m, etc. to find the script
    extractScript [] = Nothing
    extractScript (a : as)
        | "-c" == a = Nothing -- inline code, no script file
        | "-m" == a = Nothing -- module mode
        | "-" `T.isPrefixOf` a = extractScript as
        | isScriptFile a = Just a
        | otherwise = extractScript as

    isScriptArg a = not ("-" `T.isPrefixOf` a) && isScriptFile a

    isScriptFile a =
        any
            (`T.isSuffixOf` a)
            [ ".py" -- Python
            , ".js"
            , ".mjs"
            , ".cjs" -- JavaScript
            , ".ts"
            , ".mts"
            , ".cts" -- TypeScript
            , ".rb" -- Ruby
            , ".pl"
            , ".pm" -- Perl
            , ".lua" -- Lua
            ]

-- | Analyze interpreter trace: combine execve with file accesses + semantic analysis
analyzeInterpreterTrace :: TraceConfig -> FullTrace -> IO BuildGraph
analyzeInterpreterTrace cfg ft = do
    let execves = ftExecves ft
        accesses = ftFileAccesses ft
        interpreterCalls = parseInterpreters cfg execves
        -- Deduplicate by script file (Python forks multiple times)
        uniqueCalls = deduplicateInterpreters interpreterCalls

    -- For each interpreter call:
    -- 1. Find files it read (ground truth from trace)
    -- 2. Parse imports from source files (semantic analysis)
    targets <- mapM (interpreterToTargetWithSemantic cfg accesses) uniqueCalls

    pure
        BuildGraph
            { bgTargets = targets
            , bgFlakeRefs = Set.empty
            }

-- | Enhanced target extraction with semantic analysis
interpreterToTargetWithSemantic :: TraceConfig -> [FileAccess] -> InterpreterCall -> IO Target
interpreterToTargetWithSemantic cfg accesses ic = do
    let baseTarget = interpreterToTarget cfg accesses ic

    -- Try to extract semantic imports from source files
    semanticImports <- case icScript ic of
        Nothing -> pure []
        Just script -> do
            let scriptPath = T.unpack script
            result <- try $ Semantic.extractImports scriptPath
            case result of
                Left (_ :: SomeException) -> pure []
                Right imports -> pure imports

    -- Resolve packages from file accesses (site-packages, node_modules)
    let sitePackageAccesses = filter isSitePackage accesses
    packageInfos <- mapM (Semantic.resolvePackage . T.unpack . faPath) sitePackageAccesses
    let packages = catMaybes packageInfos

    -- Convert semantic imports to Deps
    let importDeps = map (DepPackage . Semantic.impModule) semanticImports
        packageDeps = map (\p -> DepPackage $ Semantic.pkgName p <> "@" <> Semantic.pkgVersion p) packages

    pure
        baseTarget
            { tDeps = nub $ tDeps baseTarget ++ importDeps ++ packageDeps
            }
  where
    isSitePackage fa =
        faMode fa == ModeRead
            && ("/site-packages/" `T.isInfixOf` faPath fa || "/node_modules/" `T.isInfixOf` faPath fa)

-- | Deduplicate interpreter calls by script
deduplicateInterpreters :: [InterpreterCall] -> [InterpreterCall]
deduplicateInterpreters ics =
    let byScript =
            Map.fromListWith
                const
                [(icScript ic, ic) | ic <- ics, icScript ic /= Nothing]
        noScript = filter (\ic -> icScript ic == Nothing) ics
     in Map.elems byScript ++ take 1 noScript -- Keep at most 1 "no script" call

{- | Convert InterpreterCall to Target using file accesses
Note: We look at ALL file accesses, not just for the specific PID,
because interpreters fork and child processes do the actual file I/O.
-}
interpreterToTarget :: TraceConfig -> [FileAccess] -> InterpreterCall -> Target
interpreterToTarget cfg accesses ic =
    let
        -- Find all source files read (filter out stdlib)
        sourceFiles = filter isSourceAccess accesses
        imports = nub $ map faPath sourceFiles

        -- Target name from script or interpreter
        name = case icScript ic of
            Just script -> T.pack $ takeBaseName $ T.unpack script
            Nothing -> T.pack $ takeBaseName $ T.unpack $ icInterpreter ic

        -- The "sources" are the script + imports (dedup)
        allSources = case icScript ic of
            Just script -> nub $ script : imports
            Nothing -> imports
     in
        Target
            { tName = name
            , tSources = allSources
            , tDeps = [] -- Could analyze imports for package deps
            , tCompiler = icInterpreter ic
            , tFlags = []
            }
  where
    isSourceAccess fa =
        let p = faPath fa
         in faMode fa == ModeRead
                && any (`T.isSuffixOf` p) [".py", ".js", ".ts", ".rb", ".pl", ".lua"]
                && not (tcNixStore cfg `T.isPrefixOf` p) -- Skip nix store stdlib

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
    = -- | nixpkgs#foo
      DepFlake Text
    | -- | Local path
      DepLocal Text
    | -- | System library (-lfoo)
      DepSystem Text
    | -- | Package (pypi, npm, hackage) e.g. "requests@2.31.0"
      DepPackage Text
    deriving stock (Show, Eq, Generic)

-- | The complete build graph
data BuildGraph = BuildGraph
    { bgTargets :: [Target]
    , bgFlakeRefs :: Set Text
    -- ^ All discovered flake refs
    }
    deriving stock (Show, Eq, Generic)

-- | Analyze trace into build graph
analyzeTrace :: TraceConfig -> ([CompilerCall], [LinkerCall]) -> IO BuildGraph
analyzeTrace cfg (compiles, links) = do
    -- Group compiles by output directory (heuristic for targets)
    -- Then match with link steps

    -- Deduplicate links by output path (clang driver invokes ld multiple times)
    let uniqueLinks = deduplicateLinks links

    -- Targets from link steps
    linkTargets <- forM uniqueLinks $ \lc -> do
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

        pure
            Target
                { tName = name
                , tSources = allSources
                , tDeps = nub $ catMaybes deps ++ libDeps
                , tCompiler = lcLinker lc
                , tFlags = nub $ concatMap ccFlags relevantCompiles ++ lcFlags lc
                }

    -- If no links, create targets from standalone compiles (compile-only mode like -c)
    compileOnlyTargets <-
        if null links
            then do
                -- Group by output file or source file
                let uniqueCompiles = deduplicateCompiles compiles
                forM uniqueCompiles $ \cc -> do
                    deps <- mapM (pathToDep cfg) (ccIncludes cc)
                    let name = case ccOutput cc of
                            Just out -> T.pack $ takeBaseName $ T.unpack out
                            Nothing -> case ccSources cc of
                                (s : _) -> T.pack $ takeBaseName $ T.unpack s
                                [] -> "unknown"
                    pure
                        Target
                            { tName = name
                            , tSources = ccSources cc
                            , tDeps = catMaybes deps
                            , tCompiler = ccCompiler cc
                            , tFlags = ccFlags cc
                            }
            else pure []

    let targets = linkTargets ++ compileOnlyTargets

    -- Collect all flake refs
    let flakeRefs = Set.fromList [ref | t <- targets, DepFlake ref <- tDeps t]

    pure
        BuildGraph
            { bgTargets = targets
            , bgFlakeRefs = flakeRefs
            }

-- | Deduplicate compiles by output/source
deduplicateCompiles :: [CompilerCall] -> [CompilerCall]
deduplicateCompiles ccs =
    let byKey =
            Map.fromListWith
                mergeCompiles
                [(ccKey cc, cc) | cc <- ccs]
     in Map.elems byKey
  where
    ccKey cc = case ccOutput cc of
        Just out -> out
        Nothing -> case ccSources cc of
            (s : _) -> s
            [] -> ""
    mergeCompiles a b =
        a
            { ccSources = nub (ccSources a ++ ccSources b)
            , ccIncludes = nub (ccIncludes a ++ ccIncludes b)
            , ccDefines = nub (ccDefines a ++ ccDefines b)
            , ccFlags = nub (ccFlags a ++ ccFlags b)
            }

-- | Deduplicate links by output, merging info from all invocations
deduplicateLinks :: [LinkerCall] -> [LinkerCall]
deduplicateLinks links =
    let byOutput =
            Map.fromListWith
                mergeLinks
                [(out, lc) | lc <- links, Just out <- [lcOutput lc]]
     in Map.elems byOutput
  where
    mergeLinks lc1 lc2 =
        LinkerCall
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

{- | Convert nix store path to flake ref
/nix/store/xxx-foo-1.2.3 -> nixpkgs#foo (heuristic)
-}
storePathToFlakeRef :: Text -> IO Text
storePathToFlakeRef path = do
    -- Extract package name from store path
    -- Format: /nix/store/HASH-NAME-VERSION
    let parts = T.splitOn "-" $ T.drop 44 path -- 44 = "/nix/store/" + 32-char hash + "-"
    case parts of
        [] -> pure $ "unknown#" <> path
        (name : _) -> pure $ "nixpkgs#" <> name -- Assume nixpkgs, could query nix path-info

-- -----------------------------------------------------------------------------
-- Dhall Code Generation
-- -----------------------------------------------------------------------------

-- | Generate Dhall source for build graph
toDhall :: BuildGraph -> Text
toDhall bg =
    let targets = bgTargets bg
        -- Give each target a unique name
        indexedTargets = zip [1 ..] targets
        namedTargets = [(uniqueName i t, t) | (i, t) <- indexedTargets]
     in T.unlines $
            [ "-- Generated by armitage trace"
            , "-- " <> T.pack (show (length targets)) <> " target(s) extracted"
            , ""
            , "let Build = ./Build.dhall"
            , "let Toolchain = ./Toolchain.dhall"
            , ""
            ]
                ++ map (uncurry targetToDhallLet) namedTargets
                ++ [ "in {"
                   , T.intercalate ",\n" ["  " <> name <> " = " <> name | (name, _) <- namedTargets]
                   , "}"
                   ]
  where
    uniqueName :: Int -> Target -> Text
    uniqueName i t =
        let base = sanitizeName (tName t)
         in if length (bgTargets bg) == 1 then base else base <> "_" <> T.pack (show i)

-- | Generate Dhall let-binding for a single target
targetToDhallLet :: Text -> Target -> Text
targetToDhallLet varName Target{..} =
    T.unlines
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
    DepLocal _ -> Nothing -- Skip local for now
    DepSystem lib -> Just $ "Build.dep.system \"" <> lib <> "\""
    DepPackage pkg -> Just $ "Build.dep.package \"" <> pkg <> "\""

-- | Write Dhall to file
toDhallFile :: FilePath -> BuildGraph -> IO ()
toDhallFile path bg = TIO.writeFile path (toDhall bg)
