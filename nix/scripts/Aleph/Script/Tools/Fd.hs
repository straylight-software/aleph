{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Fd
Description : Typed wrapper for fd (find alternative)

fd is a fast and user-friendly alternative to find. It searches for
entries in the filesystem matching a pattern.

@
import Aleph.Script
import qualified Aleph.Script.Tools.Fd as Fd

main = script $ do
  -- Find all Haskell files
  files <- Fd.fd Fd.defaults
    { Fd.extension = Just "hs"
    , Fd.type_ = Just Fd.File
    } Nothing ["."]
  mapM_ (echo . pack) files
@
-}
module Aleph.Script.Tools.Fd (
    -- * Options
    Options (..),
    defaults,

    -- * File types
    FileType (..),

    -- * Invocation
    fd,
    fd_,

    -- * Common patterns
    search,
    findFiles,
    findDirs,
    findByExt,
) where

import Aleph.Script hiding (FilePath)
import Data.Maybe (catMaybes)
import Data.Text (Text)
import qualified Data.Text as T
import Prelude (Bool (..), Eq, FilePath, Int, Maybe (..), Show, filter, fmap, map, not, return, show, ($), (++), (.), (<>))

-- | File type filter
data FileType
    = -- | Regular files (f)
      File
    | -- | Directories (d)
      Directory
    | -- | Symbolic links (l)
      Symlink
    | -- | Sockets (s)
      Socket
    | -- | Named pipes/FIFOs (p)
      Pipe
    | -- | Block devices (b)
      BlockDevice
    | -- | Character devices (c)
      CharDevice
    | -- | Executable files (x)
      Executable
    | -- | Empty files or directories (e)
      Empty
    deriving (Show, Eq)

fileTypeToArg :: FileType -> Text
fileTypeToArg = \case
    File -> "f"
    Directory -> "d"
    Symlink -> "l"
    Socket -> "s"
    Pipe -> "p"
    BlockDevice -> "b"
    CharDevice -> "c"
    Executable -> "x"
    Empty -> "e"

{- | fd options

Use 'defaults' and override fields as needed:

> defaults { hidden = True, extension = Just "hs" }
-}
data Options = Options
    { -- \** Search behavior
      hidden :: Bool
    -- ^ -H: Include hidden files/directories
    , noIgnore :: Bool
    -- ^ -I: Don't respect .gitignore, .ignore, etc.
    , noIgnoreVcs :: Bool
    -- ^ --no-ignore-vcs: Don't respect .gitignore
    , noIgnoreParent :: Bool
    -- ^ --no-ignore-parent: Don't respect ignores in parent dirs
    , unrestricted :: Bool
    -- ^ -u: Unrestricted search (hidden + no-ignore)
    , caseSensitive :: Bool
    -- ^ -s: Case-sensitive search
    , ignoreCase :: Bool
    -- ^ -i: Case-insensitive search
    , glob :: Bool
    -- ^ -g: Use glob pattern instead of regex
    , fixedStrings :: Bool
    -- ^ -F: Treat pattern as literal string
    , follow :: Bool
    -- ^ -L: Follow symlinks
    , fullPath :: Bool
    -- ^ -p: Match pattern against full path
    , absolutePath :: Bool
    -- ^ -a: Show absolute paths
    , -- \** Filtering
      type_ :: Maybe FileType
    -- ^ -t: Filter by file type
    , extension :: Maybe Text
    -- ^ -e: Filter by extension
    , exclude :: Maybe Text
    -- ^ -E: Exclude pattern (glob)
    , maxDepth :: Maybe Int
    -- ^ -d: Maximum directory depth
    , minDepth :: Maybe Int
    -- ^ --min-depth: Minimum depth
    , exactDepth :: Maybe Int
    -- ^ --exact-depth: Exact depth
    , maxResults :: Maybe Int
    -- ^ --max-results: Limit results
    , andPattern :: Maybe Text
    -- ^ --and: Additional required pattern
    , -- \** Output control
      listDetails :: Bool
    -- ^ -l: Detailed listing (like ls -l)
    , print0 :: Bool
    -- ^ -0: Null-separated output
    , color :: Maybe Text
    -- ^ -c: Color mode (auto/always/never)
    , quiet :: Bool
    -- ^ -q: Don't print, just check existence
    , showErrors :: Bool
    -- ^ --show-errors: Show filesystem errors
    , prune :: Bool
    -- ^ --prune: Don't descend into matching dirs
    , -- \** Execution
      threads :: Maybe Int
    -- ^ -j: Number of threads
    , baseDirectory :: Maybe FilePath
    -- ^ --base-directory: Change working directory
    , pathSeparator :: Maybe Text
    -- ^ --path-separator: Custom path separator
    , oneFileSystem :: Bool
    -- ^ --one-file-system: Don't cross filesystem boundaries
    }
    deriving (Show, Eq)

-- | Default options - minimal flags, let fd use its defaults
defaults :: Options
defaults =
    Options
        { hidden = False
        , noIgnore = False
        , noIgnoreVcs = False
        , noIgnoreParent = False
        , unrestricted = False
        , caseSensitive = False
        , ignoreCase = False
        , glob = False
        , fixedStrings = False
        , follow = False
        , fullPath = False
        , absolutePath = False
        , type_ = Nothing
        , extension = Nothing
        , exclude = Nothing
        , maxDepth = Nothing
        , minDepth = Nothing
        , exactDepth = Nothing
        , maxResults = Nothing
        , andPattern = Nothing
        , listDetails = False
        , print0 = False
        , color = Nothing
        , quiet = False
        , showErrors = False
        , prune = False
        , threads = Nothing
        , baseDirectory = Nothing
        , pathSeparator = Nothing
        , oneFileSystem = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag hidden "-H"
        , flag noIgnore "-I"
        , flag noIgnoreVcs "--no-ignore-vcs"
        , flag noIgnoreParent "--no-ignore-parent"
        , flag unrestricted "-u"
        , flag caseSensitive "-s"
        , flag ignoreCase "-i"
        , flag glob "-g"
        , flag fixedStrings "-F"
        , flag follow "-L"
        , flag fullPath "-p"
        , flag absolutePath "-a"
        , opt (fmap fileTypeToArg type_) "-t"
        , opt extension "-e"
        , opt exclude "-E"
        , optShow maxDepth "-d"
        , optShow minDepth "--min-depth"
        , optShow exactDepth "--exact-depth"
        , optShow maxResults "--max-results"
        , opt andPattern "--and"
        , flag listDetails "-l"
        , flag print0 "-0"
        , opt color "-c"
        , flag quiet "-q"
        , flag showErrors "--show-errors"
        , flag prune "--prune"
        , optShow threads "-j"
        , opt (fmap pack baseDirectory) "--base-directory"
        , opt pathSeparator "--path-separator"
        , flag oneFileSystem "--one-file-system"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

{- | Run fd with options, optional pattern, and paths

Returns list of matching paths. Throws on non-zero exit.
-}
fd :: Options -> Maybe Text -> [FilePath] -> Sh [FilePath]
fd opts mPattern paths = do
    let args = buildArgs opts ++ maybe [] (: []) mPattern ++ map pack paths
    output <- run "fd" args
    return $ map unpack $ filter (not . T.null) $ T.lines output

-- | Run fd, ignoring output (useful with -q for existence check)
fd_ :: Options -> Maybe Text -> [FilePath] -> Sh ()
fd_ opts mPattern paths = do
    let args = buildArgs opts ++ maybe [] (: []) mPattern ++ map pack paths
    run_ "fd" args

-- | Simple search: pat in paths with default options
search :: Text -> [FilePath] -> Sh [FilePath]
search pat = fd defaults (Just pat)

-- | Find only files matching pattern
findFiles :: Text -> [FilePath] -> Sh [FilePath]
findFiles pat = fd defaults{type_ = Just File} (Just pat)

-- | Find only directories matching pattern
findDirs :: Text -> [FilePath] -> Sh [FilePath]
findDirs pat = fd defaults{type_ = Just Directory} (Just pat)

-- | Find files by extension
findByExt :: Text -> [FilePath] -> Sh [FilePath]
findByExt ext = fd defaults{extension = Just ext} Nothing
