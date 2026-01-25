{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module      : Aleph.Script
Description : Batteries-included shell scripting for Weyl

A curated prelude for shell scripting that combines:
  - Shelly's solid foundation (thread-safe, tracing, good errors)
  - Turtle's ergonomics (streaming, format, patterns)
  - Weyl-specific helpers (GPU, containers, Nix)

@
{\-# LANGUAGE OverloadedStrings #-\}
import Aleph.Script

main :: IO ()
main = script $ do
  echo "Hello from Weyl!"
  files <- ls "."
  for_ files $ \f -> echo $ "Found: " <> toText f
@
-}
module Aleph.Script (
    -- * Running scripts
    script,
    scriptV, -- verbose (shows commands)
    scriptDebug, -- very verbose (shows output too)
    Sh,
    liftIO,

    -- * Text operations
    Text,
    pack,
    unpack,
    strip,
    stripStart,
    stripEnd,
    lines,
    unlines,
    words,
    unwords,
    intercalate,
    isPrefixOf,
    isSuffixOf,
    isInfixOf,
    replace,
    splitOn,
    breakOn,
    toLower,
    toUpper,
    T.concat,
    T.take,
    T.drop,
    T.length,
    T.null,
    T.empty,
    T.singleton,
    T.replicate,

    -- * Type-safe formatting (from Turtle)
    Format,
    format,
    (%),
    makeFormat,
    s, -- Text
    d, -- Int (decimal)
    f, -- Double (float)
    w, -- Show a => a
    fp, -- FilePath
    e, -- Scientific notation
    g, -- General number
    x, -- Hex
    o, -- Octal
    b, -- Binary
    su, -- String (unpacked)

    -- * FilePath operations
    FilePath,
    (</>),
    (<.>),
    fromText,
    toText,
    toTextIgnore,
    filename,
    dirname,
    basename,
    extension,
    parent,
    hasExtension,
    dropExtension,
    addExtension,
    replaceExtension,
    splitDirectories,
    takeFileName,
    takeDirectory,

    -- * File system operations
    ls,
    lsT, -- ls returning Text
    lsRecursive, -- recursive ls
    cp,
    cpRecursive,
    mv,
    rm,
    rmRecursive,
    mkdir,
    mkdirP, -- mkdir -p
    pwd,
    cd,
    home,
    withTmpDir,
    withTmpFile,
    symlink,
    readlink,
    canonicalize,
    touch,
    chmod,
    test_f, -- file exists?
    test_d, -- directory exists?
    test_e, -- path exists?
    test_s, -- file non-empty?
    isExecutable,

    -- * Running commands
    run, -- run, capture stdout
    run_, -- run, ignore stdout
    runFold, -- run, fold over output lines
    bash, -- run bash -c "..."
    bash_, -- run bash -c "...", ignore output
    which,
    whichAll,

    -- * Command output handling
    ExitCode (..),
    lastExitCode,
    exitCode, -- get exit code of last command
    errExit, -- control error-on-failure
    escaping, -- control shell escaping
    silently, -- suppress output
    verbosely, -- show commands
    tracing, -- control tracing

    -- * Text I/O
    echo, -- print to stdout
    echoErr, -- print to stderr
    printf, -- formatted print
    printfErr, -- formatted print to stderr
    die, -- print error and exit
    exit, -- exit with code

    -- * Environment variables
    getEnv,
    getEnvDefault,
    setEnv,
    unsetEnv,
    withEnv, -- temporarily set env var

    -- * Streaming (inspired by Turtle)
    Stream,
    stream, -- create stream from Sh [a]
    fold, -- fold a stream
    foldM, -- monadic fold
    drain, -- run stream, ignore results
    single, -- stream of one element
    select, -- stream from list
    cat, -- concatenate streams
    limit, -- take n elements
    filter_, -- filter stream
    mapStream_, -- map over stream for effects

    -- * Common folds
    Fold,
    Fold.list,
    Fold.head,
    Fold.last,
    foldNull,
    foldLength,
    Fold.all,
    Fold.any,
    Fold.sum,
    Fold.product,
    Fold.mconcat,
    Fold.foldMap,
    Fold.elem,
    Fold.find,

    -- * Control flow
    when,
    unless,
    void,
    forM_,
    M.mapM_,
    forM,
    replicateM,
    replicateM_,
    guard,
    msum,
    filterM,

    -- * Error handling
    try,
    tryIO,
    catch,
    catchIO,
    bracket,
    finally,
    onException,
    throwM,
    Exception,
    SomeException,
    IOException,

    -- * Maybe / Either
    Maybe (..),
    maybe,
    fromMaybe,
    isJust,
    isNothing,
    maybeToList,
    listToMaybe,
    catMaybes,
    mapMaybe,
    Either (..),
    either,
    isLeft,
    isRight,
    fromLeft,
    fromRight,
    lefts,
    rights,
    partitionEithers,

    -- * JSON
    Value (..),
    Object,
    Array,
    decode,
    decode',
    eitherDecode,
    encode,
    (.:),
    (.:?),
    (.:!),
    (.=),
    object,
    ToJSON (..),
    FromJSON (..),

    -- * ByteString
    ByteString,
    LByteString,
    encodeUtf8,
    decodeUtf8,
    decodeUtf8',

    -- * Time
    UTCTime,
    getCurrentTime,
    diffUTCTime,
    NominalDiffTime,
    sleep, -- sleep n seconds
    sleepMs, -- sleep n milliseconds
    timed, -- time an action

    -- * Concurrency
    async,
    wait,
    cancel,
    concurrently,
    race,
    Async,
    MVar,
    newMVar,
    readMVar,
    modifyMVar,
    modifyMVar_,

    -- * Retry / resilience
    retry,
    retryWithBackoff,
    timeout,

    -- * Useful operators
    (&),
    (<&>),
    (<|>),
    (<=<),
    (>=>),

    -- * Weyl-specific: GPU
    GpuArch (..),
    gpuCapability,
    gpuArchName,
    detectGpu,
    withGpuBinds,

    -- * Weyl-specific: Nix
    inNixShell,
    nixStorePath,
    nixCurrentSystem,
) where

import Prelude hiding (FilePath, lines, unlines, unwords, words)
import qualified Prelude

import Shelly hiding (
    FilePath,
    -- Hide things we redefine with better types

    bash,
    bash_,
    canonicalize,
    cd,
    cp,
    cp_r,
    echo,
    errExit,
    escaping,
    exit,
    find,
    fromText,
    get_env,
    get_env_text,
    ls,
    lsT,
    mkdir,
    mkdir_p,
    mv,
    pwd,
    rm,
    rm_rf,
    run,
    run_,
    silently,
    sleep,
    test_d,
    test_e,
    test_f,
    test_s,
    toTextIgnore,
    trace,
    tracing,
    unless,
    verbosely,
    when,
    which,
    withTmpDir,
    (<.>),
    (</>),
 )
import qualified Shelly as S
import qualified Shelly.Lifted

import Data.ByteString (ByteString)
import qualified Data.ByteString.Lazy as LBS
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.IO as TIO

import qualified System.Directory as Dir
import System.FilePath ((<.>), (</>))
import qualified System.FilePath as FP

import Data.Aeson (
    Array,
    FromJSON (..),
    Object,
    ToJSON (..),
    Value (..),
    decode,
    decode',
    eitherDecode,
    encode,
    object,
    (.:),
    (.:!),
    (.:?),
    (.=),
 )

import Control.Concurrent (MVar, modifyMVar, modifyMVar_, newMVar, readMVar, threadDelay)
import Control.Concurrent.Async (Async, async, cancel, concurrently, race, wait)
import Control.Exception (Exception, IOException, SomeException, throwIO)
import qualified Control.Exception as E
import Control.Monad hiding (foldM, mapM_)
import qualified Control.Monad as M
import Control.Monad.IO.Class (MonadIO, liftIO)
import Data.Either
import Data.Maybe
import qualified System.Timeout as Timeout

import Control.Applicative ((<|>))
import Data.Function ((&))
import Data.Functor ((<&>))
import Data.Time (NominalDiffTime, UTCTime, diffUTCTime, getCurrentTime)

import Control.Foldl (Fold)
import qualified Control.Foldl as Fold

import qualified System.Environment as Env
import System.Exit (ExitCode (..))
import qualified System.IO as IO

-- ============================================================================
-- FilePath (use System.FilePath, not Shelly's)
-- ============================================================================

-- | FilePath is just String (standard Haskell)
type FilePath = Prelude.FilePath

-- | Convert Text to FilePath
fromText :: Text -> FilePath
fromText = T.unpack

-- | Convert FilePath to Text (may fail on invalid unicode)
toText :: FilePath -> Either String Text
toText fp = Right (T.pack fp)

-- | Convert FilePath to Text (replace invalid chars)
toTextIgnore :: FilePath -> Text
toTextIgnore = T.pack

-- | Get filename from path
filename :: FilePath -> FilePath
filename = FP.takeFileName

-- | Get directory from path
dirname :: FilePath -> FilePath
dirname = FP.takeDirectory

-- | Get filename without extension
basename :: FilePath -> FilePath
basename = FP.takeBaseName

-- | Get extension (with dot)
extension :: FilePath -> String
extension = FP.takeExtension

-- | Get parent directory
parent :: FilePath -> FilePath
parent = FP.takeDirectory

-- | Check if path has extension
hasExtension :: FilePath -> Bool
hasExtension = FP.hasExtension

-- | Drop extension from path
dropExtension :: FilePath -> FilePath
dropExtension = FP.dropExtension

-- | Add extension to path
addExtension :: FilePath -> String -> FilePath
addExtension = FP.addExtension

-- | Replace extension
replaceExtension :: FilePath -> String -> FilePath
replaceExtension = FP.replaceExtension

-- | Split path into components
splitDirectories :: FilePath -> [FilePath]
splitDirectories = FP.splitDirectories

-- | Alias for filename
takeFileName :: FilePath -> FilePath
takeFileName = FP.takeFileName

-- | Alias for dirname
takeDirectory :: FilePath -> FilePath
takeDirectory = FP.takeDirectory

-- ============================================================================
-- Text operations (re-exports from Data.Text)
-- ============================================================================

pack :: String -> Text
pack = T.pack

unpack :: Text -> String
unpack = T.unpack

strip :: Text -> Text
strip = T.strip

stripStart :: Text -> Text
stripStart = T.stripStart

stripEnd :: Text -> Text
stripEnd = T.stripEnd

lines :: Text -> [Text]
lines = T.lines

unlines :: [Text] -> Text
unlines = T.unlines

words :: Text -> [Text]
words = T.words

unwords :: [Text] -> Text
unwords = T.unwords

intercalate :: Text -> [Text] -> Text
intercalate = T.intercalate

isPrefixOf :: Text -> Text -> Bool
isPrefixOf = T.isPrefixOf

isSuffixOf :: Text -> Text -> Bool
isSuffixOf = T.isSuffixOf

isInfixOf :: Text -> Text -> Bool
isInfixOf = T.isInfixOf

replace :: Text -> Text -> Text -> Text
replace = T.replace

splitOn :: Text -> Text -> [Text]
splitOn = T.splitOn

breakOn :: Text -> Text -> (Text, Text)
breakOn = T.breakOn

toLower :: Text -> Text
toLower = T.toLower

toUpper :: Text -> Text
toUpper = T.toUpper

-- ============================================================================
-- Type-safe formatting (inspired by Turtle.Format)
-- ============================================================================

-- | A format string that takes arguments of type @a@ and produces @Text@
newtype Format a = Format {runFormat :: a -> Text}

instance Semigroup (Format a) where
    Format f <> Format g = Format $ \a -> f a <> g a

instance Monoid (Format a) where
    mempty = Format $ const ""

-- | Format a value using a format string
format :: Format a -> a -> Text
format = runFormat

-- | Concatenate two format strings
(%) :: Format b -> Format a -> Format (b, a)
Format f % Format g = Format $ \(b, a) -> f b <> g a

infixr 9 %

-- | Create a format from a function
makeFormat :: (a -> Text) -> Format a
makeFormat = Format

-- | Format Text (identity)
s :: Format Text
s = Format id

-- | Format Int as decimal
d :: Format Int
d = Format (T.pack . show)

-- | Format Double as float
f :: Format Double
f = Format (T.pack . show)

-- | Format any Show instance
w :: (Show a) => Format a
w = Format (T.pack . show)

-- | Format FilePath
fp :: Format FilePath
fp = Format T.pack

-- | Format in scientific notation
e :: Format Double
e = Format (T.pack . show) -- TODO: proper scientific

-- | Format number (general)
g :: Format Double
g = Format (T.pack . show)

-- | Format Int as hexadecimal
x :: Format Int
x = Format toHex
  where
    toHex n
        | n < 0 = "-" <> toHex (abs n)
        | n < 16 = T.singleton (hexDigit n)
        | otherwise = toHex (n `div` 16) <> T.singleton (hexDigit (n `mod` 16))
    hexDigit d = "0123456789abcdef" !! d

-- | Format Int as octal
o :: Format Int
o = Format toOct
  where
    toOct n
        | n < 0 = "-" <> toOct (abs n)
        | n < 8 = T.pack (show n)
        | otherwise = toOct (n `div` 8) <> T.pack (show (n `mod` 8))

-- | Format Int as binary
b :: Format Int
b = Format toBinary
  where
    toBinary n
        | n < 0 = "-" <> toBinary (abs n)
        | n < 2 = T.pack (show n)
        | otherwise = toBinary (n `div` 2) <> T.pack (show (n `mod` 2))

-- | Format String (unpack from Text)
su :: Format String
su = Format T.pack

-- ============================================================================
-- Running scripts
-- ============================================================================

-- | Run a script (silent, errors on failure)
script :: Sh a -> IO a
script = shelly

-- | Run a script verbosely (shows commands)
scriptV :: Sh a -> IO a
scriptV = shelly . verbosely

-- | Run a script with full debug output
scriptDebug :: Sh a -> IO a
scriptDebug = shelly . verbosely . S.tracing True

-- ============================================================================
-- File system operations
-- ============================================================================

-- | List directory contents
ls :: FilePath -> Sh [FilePath]
ls p = map (T.unpack . S.toTextIgnore) <$> S.ls (S.fromText $ T.pack p)

-- | List directory contents as Text
lsT :: FilePath -> Sh [Text]
lsT p = map S.toTextIgnore <$> S.ls (S.fromText $ T.pack p)

-- | Recursively list directory
lsRecursive :: FilePath -> Sh [FilePath]
lsRecursive p = map (T.unpack . S.toTextIgnore) <$> S.findWhen (const $ pure True) (S.fromText $ T.pack p)

-- | Copy file
cp :: FilePath -> FilePath -> Sh ()
cp src dst = S.cp (S.fromText $ T.pack src) (S.fromText $ T.pack dst)

-- | Copy directory recursively
cpRecursive :: FilePath -> FilePath -> Sh ()
cpRecursive src dst = S.cp_r (S.fromText $ T.pack src) (S.fromText $ T.pack dst)

-- | Move file or directory
mv :: FilePath -> FilePath -> Sh ()
mv src dst = S.mv (S.fromText $ T.pack src) (S.fromText $ T.pack dst)

-- | Remove file
rm :: FilePath -> Sh ()
rm p = S.rm (S.fromText $ T.pack p)

-- | Remove directory recursively
rmRecursive :: FilePath -> Sh ()
rmRecursive p = S.rm_rf (S.fromText $ T.pack p)

-- | Create directory
mkdir :: FilePath -> Sh ()
mkdir p = S.mkdir (S.fromText $ T.pack p)

-- | Create directory and parents (mkdir -p)
mkdirP :: FilePath -> Sh ()
mkdirP p = S.mkdir_p (S.fromText $ T.pack p)

-- | Get current working directory
pwd :: Sh FilePath
pwd = T.unpack . S.toTextIgnore <$> S.pwd

-- | Change working directory
cd :: FilePath -> Sh ()
cd p = S.cd (S.fromText $ T.pack p)

-- | Get home directory
home :: Sh FilePath
home = liftIO Dir.getHomeDirectory

-- | Run action in temporary directory (auto-cleanup)
withTmpDir :: (FilePath -> Sh a) -> Sh a
withTmpDir f = S.withTmpDir $ \d -> f (T.unpack $ S.toTextIgnore d)

-- | Run action with temporary file (auto-cleanup)
withTmpFile :: (FilePath -> Sh a) -> Sh a
withTmpFile f = withTmpDir $ \dir -> do
    let path = dir </> "tmp"
    liftIO $ Prelude.writeFile path ""
    f path

-- | Create symbolic link
symlink :: FilePath -> FilePath -> Sh ()
symlink target link = run_ "ln" ["-s", T.pack target, T.pack link]

-- | Read symbolic link target
readlink :: FilePath -> Sh FilePath
readlink p = T.unpack . strip <$> run "readlink" ["-f", T.pack p]

-- | Canonicalize path (resolve symlinks)
canonicalize :: FilePath -> Sh FilePath
canonicalize = readlink

-- | Touch a file (create if missing, update mtime if exists)
touch :: FilePath -> Sh ()
touch p = run_ "touch" [T.pack p]

-- | Change file permissions (octal mode string, e.g. "755")
chmod :: Text -> FilePath -> Sh ()
chmod mode p = run_ "chmod" [mode, T.pack p]

-- | Test if file exists
test_f :: FilePath -> Sh Bool
test_f p = S.test_f (S.fromText $ T.pack p)

-- | Test if directory exists
test_d :: FilePath -> Sh Bool
test_d p = S.test_d (S.fromText $ T.pack p)

-- | Test if path exists (file or directory)
test_e :: FilePath -> Sh Bool
test_e p = S.test_e (S.fromText $ T.pack p)

-- | Test if file is non-empty
test_s :: FilePath -> Sh Bool
test_s p = S.test_s (S.fromText $ T.pack p)

-- | Test if file is executable
isExecutable :: FilePath -> Sh Bool
isExecutable p = liftIO $ Dir.executable <$> Dir.getPermissions p

-- ============================================================================
-- Running commands
-- ============================================================================

-- | Run command, capture stdout
run :: FilePath -> [Text] -> Sh Text
run cmd args = S.run (S.fromText $ T.pack cmd) args

-- | Run command, ignore stdout
run_ :: FilePath -> [Text] -> Sh ()
run_ cmd args = S.run_ (S.fromText $ T.pack cmd) args

-- | Run command, fold over output lines
runFold :: FilePath -> [Text] -> Fold Text a -> Sh a
runFold cmd args fld = do
    out <- run cmd args
    pure $ Fold.fold fld (T.lines out)

-- | Run bash command
bash :: Text -> Sh Text
bash cmd = S.run "bash" ["-c", cmd]

-- | Run bash command, ignore output
bash_ :: Text -> Sh ()
bash_ cmd = S.run_ "bash" ["-c", cmd]

-- | Find executable in PATH
which :: FilePath -> Sh (Maybe FilePath)
which cmd = do
    result <- S.which (S.fromText $ T.pack cmd)
    pure $ (T.unpack . S.toTextIgnore) <$> result

-- | Find all matching executables in PATH
whichAll :: FilePath -> Sh [FilePath]
whichAll cmd =
    map (T.unpack . S.toTextIgnore)
        <$> S.findWhen (const $ pure True) (S.fromText "/usr/bin") -- TODO: proper PATH search

-- ============================================================================
-- Command output handling
-- ============================================================================

-- | Get exit code of last command
exitCode :: Sh Int
exitCode = S.lastExitCode

-- | Control whether errors cause script to fail
errExit :: Bool -> Sh a -> Sh a
errExit = S.errExit

-- | Control shell escaping
escaping :: Bool -> Sh a -> Sh a
escaping = S.escaping

-- | Suppress all output
silently :: Sh a -> Sh a
silently = S.silently

-- | Show all commands being run
verbosely :: Sh a -> Sh a
verbosely = S.verbosely

-- | Control command tracing
tracing :: Bool -> Sh a -> Sh a
tracing = S.tracing

-- ============================================================================
-- Text I/O
-- ============================================================================

-- | Print to stdout
echo :: Text -> Sh ()
echo = S.echo

-- | Print to stderr
echoErr :: Text -> Sh ()
echoErr msg = liftIO $ TIO.hPutStrLn IO.stderr msg

-- | Formatted print to stdout
printf :: Format a -> a -> Sh ()
printf fmt a = echo (format fmt a)

-- | Formatted print to stderr
printfErr :: Format a -> a -> Sh ()
printfErr fmt a = echoErr (format fmt a)

-- | Print error message and exit with failure
die :: Text -> Sh a
die msg = do
    echoErr $ "Error: " <> msg
    exit 1
    error "unreachable"

-- | Exit with code
exit :: Int -> Sh ()
exit 0 = S.exit 0
exit n = S.errorExit $ "Exit code: " <> T.pack (show n)

-- ============================================================================
-- Environment variables
-- ============================================================================

-- | Get environment variable (Nothing if unset)
getEnv :: Text -> Sh (Maybe Text)
getEnv = S.get_env

-- | Get environment variable with default
getEnvDefault :: Text -> Text -> Sh Text
getEnvDefault name def = fromMaybe def <$> getEnv name

-- | Set environment variable
setEnv :: Text -> Text -> Sh ()
setEnv = S.setenv

-- | Unset environment variable
unsetEnv :: Text -> Sh ()
unsetEnv name = S.setenv name ""

-- | Temporarily set environment variable
withEnv :: Text -> Text -> Sh a -> Sh a
withEnv name val action = do
    old <- getEnv name
    setEnv name val
    result <- action
    case old of
        Nothing -> unsetEnv name
        Just v -> setEnv name v
    pure result

-- ============================================================================
-- Streaming (inspired by Turtle)
-- ============================================================================

-- | A stream of values (lazy list in Sh monad)
newtype Stream a = Stream {unStream :: Sh [a]}

instance Functor Stream where
    fmap f (Stream as) = Stream $ fmap (Prelude.map f) as

instance Applicative Stream where
    pure a = Stream $ pure [a]
    Stream fs <*> Stream as = Stream $ do
        fs' <- fs
        as' <- as
        pure [f a | f <- fs', a <- as']

instance Monad Stream where
    Stream as >>= f = Stream $ do
        as' <- as
        concat <$> traverse (unStream . f) as'

-- | Create stream from list action
stream :: Sh [a] -> Stream a
stream = Stream

-- | Fold a stream
fold :: Stream a -> Fold a b -> Sh b
fold (Stream as) fld = Fold.fold fld <$> as

-- | Monadic fold over stream
foldM :: (b -> a -> Sh b) -> b -> Stream a -> Sh b
foldM f z (Stream as) = as >>= M.foldM f z

-- | Run stream for effects only
drain :: Stream a -> Sh ()
drain (Stream as) = void as

-- | Stream of one element
single :: a -> Stream a
single = pure

-- | Stream from list
select :: [a] -> Stream a
select = Stream . pure

-- | Concatenate streams
cat :: [Stream a] -> Stream a
cat streams = Stream $ concat <$> traverse unStream streams

-- | Take first n elements
limit :: Int -> Stream a -> Stream a
limit n (Stream as) = Stream $ Prelude.take n <$> as

-- | Filter stream
filter_ :: (a -> Bool) -> Stream a -> Stream a
filter_ p (Stream as) = Stream $ Prelude.filter p <$> as

-- | Map over stream for effects
mapStream_ :: (a -> Sh b) -> Stream a -> Sh ()
mapStream_ f (Stream as) = as >>= M.mapM_ f

-- | Fold.null re-exported with different name to avoid conflict
foldNull :: Fold a Bool
foldNull = Fold.null

-- | Fold.length re-exported with different name to avoid conflict
foldLength :: Fold a Int
foldLength = Fold.length

-- ============================================================================
-- Time operations
-- ============================================================================

-- | Sleep for n seconds
sleep :: Double -> Sh ()
sleep secs = liftIO $ threadDelay (round $ secs * 1000000)

-- | Sleep for n milliseconds
sleepMs :: Int -> Sh ()
sleepMs ms = liftIO $ threadDelay (ms * 1000)

-- | Time an action, return (result, duration in seconds)
timed :: Sh a -> Sh (a, Double)
timed action = do
    start <- liftIO getCurrentTime
    result <- action
    end <- liftIO getCurrentTime
    let duration = realToFrac (diffUTCTime end start)
    pure (result, duration)

-- ============================================================================
-- Error handling
-- ============================================================================

-- | Try an action, catching all exceptions
try :: (Exception e) => Sh a -> Sh (Either e a)
try action = do
    result <- liftIO $ E.try $ shelly action
    case result of
        Left e -> pure $ Left e
        Right a -> pure $ Right a

-- | Try, catching IOException
tryIO :: Sh a -> Sh (Either IOException a)
tryIO = try

-- | Catch exceptions
catch :: (Exception e) => Sh a -> (e -> Sh a) -> Sh a
catch action handler = do
    result <- try action
    case result of
        Left e -> handler e
        Right a -> pure a

-- | Catch IOException
catchIO :: Sh a -> (IOException -> Sh a) -> Sh a
catchIO = catch

-- | Bracket pattern
bracket :: Sh a -> (a -> Sh b) -> (a -> Sh c) -> Sh c
bracket acquire release action = do
    resource <- acquire
    result <-
        action resource `catch` \(e :: SomeException) -> do
            _ <- release resource
            liftIO $ throwIO e
    _ <- release resource
    pure result

-- | Finally pattern
finally :: Sh a -> Sh b -> Sh a
finally action cleanup = do
    result <-
        action `catch` \(e :: SomeException) -> do
            _ <- cleanup
            liftIO $ throwIO e
    _ <- cleanup
    pure result

-- | Run cleanup on exception
onException :: Sh a -> Sh b -> Sh a
onException action cleanup =
    action `catch` \(e :: SomeException) -> do
        _ <- cleanup
        liftIO $ throwIO e

-- | Throw an exception
throwM :: (Exception e) => e -> Sh a
throwM = liftIO . throwIO

-- ============================================================================
-- Retry / resilience
-- ============================================================================

-- | Retry an action up to n times
retry :: Int -> Sh a -> Sh a
retry n action
    | n <= 1 = action
    | otherwise = action `catch` \(_ :: SomeException) -> retry (n - 1) action

-- | Retry with exponential backoff (initial delay in ms)
retryWithBackoff :: Int -> Int -> Sh a -> Sh a
retryWithBackoff maxRetries initialDelayMs action = go 1 initialDelayMs
  where
    go attempt delayMs
        | attempt > maxRetries = action
        | otherwise =
            action `catch` \(_ :: SomeException) -> do
                echoErr $ "Retry " <> T.pack (show attempt) <> "/" <> T.pack (show maxRetries)
                sleepMs delayMs
                go (attempt + 1) (delayMs * 2)

-- | Run action with timeout (in seconds)
timeout :: Double -> Sh a -> Sh (Maybe a)
timeout secs action = liftIO $ Timeout.timeout (round $ secs * 1000000) (shelly action)

-- ============================================================================
-- ByteString
-- ============================================================================

type LByteString = LBS.ByteString

encodeUtf8 :: Text -> ByteString
encodeUtf8 = TE.encodeUtf8

decodeUtf8 :: ByteString -> Text
decodeUtf8 = TE.decodeUtf8

decodeUtf8' :: ByteString -> Either String Text
decodeUtf8' bs = case TE.decodeUtf8' bs of
    Left e -> Left (show e)
    Right t -> Right t

-- ============================================================================
-- Weyl-specific: GPU
-- ============================================================================

-- | NVIDIA GPU architectures
data GpuArch
    = -- | SM 7.0 (V100)
      Volta
    | -- | SM 7.5 (RTX 20xx, T4)
      Turing
    | -- | SM 8.0/8.6 (A100, RTX 30xx)
      Ampere
    | -- | SM 8.9 (RTX 40xx, L40)
      Ada
    | -- | SM 9.0 (H100, H200)
      Hopper
    | -- | SM 10.0/12.0 (B100, B200, GB200)
      Blackwell
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | Get compute capability for an architecture
gpuCapability :: GpuArch -> Text
gpuCapability = \case
    Volta -> "7.0"
    Turing -> "7.5"
    Ampere -> "8.0"
    Ada -> "8.9"
    Hopper -> "9.0"
    Blackwell -> "10.0"

-- | Get marketing name for an architecture
gpuArchName :: GpuArch -> Text
gpuArchName = \case
    Volta -> "volta"
    Turing -> "turing"
    Ampere -> "ampere"
    Ada -> "ada"
    Hopper -> "hopper"
    Blackwell -> "blackwell"

-- | Detect available GPU (returns architecture if nvidia-smi works)
detectGpu :: Sh (Maybe GpuArch)
detectGpu = do
    result <- errExit False $ run "nvidia-smi" ["--query-gpu=compute_cap", "--format=csv,noheader"]
    code <- exitCode
    if code == 0
        then pure $ parseCapability (strip result)
        else pure Nothing
  where
    parseCapability cap
        | "7.0" `isPrefixOf` cap = Just Volta
        | "7.5" `isPrefixOf` cap = Just Turing
        | "8.0" `isPrefixOf` cap = Just Ampere
        | "8.6" `isPrefixOf` cap = Just Ampere
        | "8.9" `isPrefixOf` cap = Just Ada
        | "9.0" `isPrefixOf` cap = Just Hopper
        | "10." `isPrefixOf` cap = Just Blackwell
        | "12." `isPrefixOf` cap = Just Blackwell
        | otherwise = Nothing

-- | Get bwrap bind arguments for GPU access
withGpuBinds :: Sh [Text]
withGpuBinds = do
    -- Find nvidia devices
    devs <- ls "/dev"
    let nvDevs = Prelude.filter (isPrefixOf "nvidia" . T.pack) devs

    -- Find DRI devices
    driExists <- test_d "/dev/dri"
    driDevs <- if driExists then ls "/dev/dri" else pure []

    let allDevs = nvDevs ++ Prelude.map ("/dev/dri" </>) driDevs
        devBinds = concatMap (\dev -> ["--dev-bind", T.pack dev, T.pack dev]) allDevs

    -- Find nvidia driver
    driverBinds <- errExit False $ do
        nvPath <- strip <$> bash "readlink -f /run/current-system/sw/bin/nvidia-smi 2>/dev/null || true"
        if T.null nvPath
            then pure []
            else do
                let driverDir = T.unpack $ replace "/bin/nvidia-smi" "" nvPath
                exists <- test_d driverDir
                if exists
                    then
                        pure
                            [ "--ro-bind"
                            , T.pack (driverDir </> "bin")
                            , "/usr/local/nvidia/bin"
                            , "--ro-bind"
                            , T.pack (driverDir </> "lib")
                            , "/usr/local/nvidia/lib64"
                            ]
                    else pure []

    -- OpenGL drivers
    gl1 <- test_d "/run/opengl-driver"
    gl2 <- test_d "/run/opengl-driver-32"
    let glBinds =
            (if gl1 then ["--ro-bind", "/run/opengl-driver", "/run/opengl-driver"] else [])
                ++ (if gl2 then ["--ro-bind", "/run/opengl-driver-32", "/run/opengl-driver-32"] else [])

    pure $ devBinds ++ driverBinds ++ glBinds ++ ["--ro-bind", "/nix/store", "/nix/store"]

-- ============================================================================
-- Nix
-- ============================================================================

-- | Check if running in nix-shell
inNixShell :: Sh Bool
inNixShell = isJust <$> getEnv "IN_NIX_SHELL"

-- | Get nix store path for a package
nixStorePath :: Text -> Sh (Maybe FilePath)
nixStorePath pkg = do
    result <- errExit False $ bash $ "nix-store -q --outputs $(which " <> pkg <> ") 2>/dev/null"
    code <- exitCode
    if code == 0
        then pure $ Just $ T.unpack $ strip result
        else pure Nothing

-- | Get current system (e.g., "x86_64-linux")
nixCurrentSystem :: Sh Text
nixCurrentSystem = do
    result <- getEnv "NIX_SYSTEM"
    case result of
        Just sys -> pure sys
        Nothing -> strip <$> bash "nix eval --raw nixpkgs#system 2>/dev/null || echo x86_64-linux"
