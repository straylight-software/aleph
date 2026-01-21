{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Ls
Description : Typed wrapper for ls

This module was auto-generated from @ls --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Ls (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    ls,
    ls_,
) where

import Aleph.Script hiding (ls)
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed.
-}
data Options = Options
    { all :: Bool
    -- ^ -a: do not ignore entries starting with .
    , almostAll :: Bool
    -- ^ -A: do not list implied . and ..
    , author :: Bool
    -- ^ with -l, print the author of each file
    , escape :: Bool
    -- ^ -b: print C-style escapes for nongraphic characters
    , blockSize :: Maybe Text
    -- ^ with -l, scale sizes by SIZE when printing them;
    , ignoreBackups :: Bool
    -- ^ -B: do not list implied entries ending with ~
    , optC :: Bool
    -- ^ -c: with -lt: sort by, and show, ctime (time of last
    , color :: Maybe Text
    -- ^ color the output WHEN; more info below
    , directory :: Bool
    -- ^ -d: list directories themselves, not their contents
    , dired :: Bool
    -- ^ -D: generate output designed for Emacs' dired mode
    , optF :: Bool
    -- ^ -f: same as -a -U
    , classify :: Maybe Text
    -- ^ -F: append indicator (one of */=>@|) to entries WHEN
    , fileType :: Bool
    -- ^ likewise, except do not append '*'
    , format :: Maybe Text
    -- ^ across,horizontal (-x), commas (-m), long (-l),
    , fullTime :: Bool
    -- ^ like -l --time-style=full-iso
    , optG :: Bool
    -- ^ -g: like -l, but do not list owner
    , groupDirectoriesFirst :: Bool
    , noGroup :: Bool
    -- ^ -G: in a long listing, don't print group names
    , humanReadable :: Bool
    -- ^ -h: with -l and -s, print sizes like 1K 234M 2G etc.
    , si :: Bool
    -- ^ likewise, but use powers of 1000 not 1024
    , dereferenceCommandLine :: Bool
    , dereferenceCommandLineSymlinkToDir :: Bool
    , hide :: Maybe Text
    -- ^ do not list implied entries matching shell PATTERN
    , hyperlink :: Maybe Text
    -- ^ hyperlink file names WHEN
    , indicatorStyle :: Maybe Text
    , inode :: Bool
    -- ^ -i: print the index number of each file
    , ignore :: Maybe Text
    -- ^ -I: do not list implied entries matching shell PATTERN
    , kibibytes :: Bool
    -- ^ -k: default to 1024-byte blocks for file system usage;
    , optL :: Bool
    -- ^ -l: use a long listing format
    , dereference :: Bool
    -- ^ -L: when showing file information for a symbolic
    , optM :: Bool
    -- ^ -m: fill width with a comma separated list of entries
    , numericUidGid :: Bool
    -- ^ -n: like -l, but list numeric user and group IDs
    , literal :: Bool
    -- ^ -N: print entry names without quoting
    , optO :: Bool
    -- ^ -o: like -l, but do not list group information
    , hideControlChars :: Bool
    -- ^ -q: print ? instead of nongraphic characters
    , showControlChars :: Bool
    -- ^ show nongraphic characters as-is (the default,
    , quoteName :: Bool
    -- ^ -Q: enclose entry names in double quotes
    , quotingStyle :: Maybe Text
    -- ^ use quoting style WORD for entry names:
    , reverse_ :: Bool
    -- ^ -r: reverse order while sorting
    , recursive :: Bool
    -- ^ -R: list subdirectories recursively
    , size :: Bool
    -- ^ -s: print the allocated size of each file, in blocks
    , optS :: Bool
    -- ^ -S: sort by file size, largest first
    , sort :: Maybe Text
    -- ^ change default 'name' sort to WORD:
    , time :: Maybe Text
    -- ^ select which timestamp used to display or sort;
    , timeStyle :: Maybe Text
    , optT :: Bool
    -- ^ -t: sort by time, newest first; see --time
    , tabsize :: Maybe Text
    -- ^ -T: assume tab stops at each COLS instead of 8
    , optU :: Bool
    -- ^ -u: with -lt: sort by, and show, access time;
    , optV :: Bool
    -- ^ -v: natural sort of (version) numbers within text
    , width :: Maybe Text
    -- ^ -w: set output width to COLS.  0 means no limit
    , optX :: Bool
    -- ^ -x: list entries by lines instead of by columns
    , context :: Bool
    -- ^ -Z: print any security context of each file
    , zero :: Bool
    -- ^ end each output line with NUL, not newline
    , opt1 :: Bool
    -- ^ -1: list one file per line
    }
    deriving (Show, Eq)

-- | Default options
defaults :: Options
defaults =
    Options
        { all = False
        , almostAll = False
        , author = False
        , escape = False
        , blockSize = Nothing
        , ignoreBackups = False
        , optC = False
        , color = Nothing
        , directory = False
        , dired = False
        , optF = False
        , classify = Nothing
        , fileType = False
        , format = Nothing
        , fullTime = False
        , optG = False
        , groupDirectoriesFirst = False
        , noGroup = False
        , humanReadable = False
        , si = False
        , dereferenceCommandLine = False
        , dereferenceCommandLineSymlinkToDir = False
        , hide = Nothing
        , hyperlink = Nothing
        , indicatorStyle = Nothing
        , inode = False
        , ignore = Nothing
        , kibibytes = False
        , optL = False
        , dereference = False
        , optM = False
        , numericUidGid = False
        , literal = False
        , optO = False
        , hideControlChars = False
        , showControlChars = False
        , quoteName = False
        , quotingStyle = Nothing
        , reverse_ = False
        , recursive = False
        , size = False
        , optS = False
        , sort = Nothing
        , time = Nothing
        , timeStyle = Nothing
        , optT = False
        , tabsize = Nothing
        , optU = False
        , optV = False
        , width = Nothing
        , optX = False
        , context = False
        , zero = False
        , opt1 = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag all "--all"
        , flag almostAll "--almost-all"
        , flag author "--author"
        , flag escape "--escape"
        , opt blockSize "--block-size"
        , flag ignoreBackups "--ignore-backups"
        , flag optC "-c"
        , opt color "--color"
        , flag directory "--directory"
        , flag dired "--dired"
        , flag optF "-f"
        , opt classify "--classify"
        , flag fileType "--file-type"
        , opt format "--format"
        , flag fullTime "--full-time"
        , flag optG "-g"
        , flag groupDirectoriesFirst "--group-directories-first"
        , flag noGroup "--no-group"
        , flag humanReadable "--human-readable"
        , flag si "--si"
        , flag dereferenceCommandLine "--dereference-command-line"
        , flag dereferenceCommandLineSymlinkToDir "--dereference-command-line-symlink-to-dir"
        , opt hide "--hide"
        , opt hyperlink "--hyperlink"
        , opt indicatorStyle "--indicator-style"
        , flag inode "--inode"
        , opt ignore "--ignore"
        , flag kibibytes "--kibibytes"
        , flag optL "-l"
        , flag dereference "--dereference"
        , flag optM "-m"
        , flag numericUidGid "--numeric-uid-gid"
        , flag literal "--literal"
        , flag optO "-o"
        , flag hideControlChars "--hide-control-chars"
        , flag showControlChars "--show-control-chars"
        , flag quoteName "--quote-name"
        , opt quotingStyle "--quoting-style"
        , flag reverse_ "--reverse"
        , flag recursive "--recursive"
        , flag size "--size"
        , flag optS "-S"
        , opt sort "--sort"
        , opt time "--time"
        , opt timeStyle "--time-style"
        , flag optT "-t"
        , opt tabsize "--tabsize"
        , flag optU "-u"
        , flag optV "-v"
        , opt width "--width"
        , flag optX "-x"
        , flag context "--context"
        , flag zero "--zero"
        , flag opt1 "-1"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

-- | Run ls with options and additional arguments
ls :: Options -> [Text] -> Sh Text
ls opts args = run "ls" (buildArgs opts ++ args)

-- | Run ls, ignoring output
ls_ :: Options -> [Text] -> Sh ()
ls_ opts args = run_ "ls" (buildArgs opts ++ args)
