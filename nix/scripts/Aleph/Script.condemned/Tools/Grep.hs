{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Grep
Description : Typed wrapper for grep

This module was auto-generated from @grep --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Grep (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    grep,
    grep_,
) where

import Aleph.Script hiding (FilePath)
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed.
-}
data Options = Options
    { extendedRegexp :: Maybe Text
    -- ^ -E: are extended regular expressions
    , fixedStrings :: Maybe Text
    -- ^ -F: are strings
    , basicRegexp :: Maybe Text
    -- ^ -G: are basic regular expressions
    , perlRegexp :: Maybe Text
    -- ^ -P: are Perl regular expressions
    , regexp :: Maybe Text
    -- ^ -e: use PATTERNS for matching
    , file :: Maybe FilePath
    -- ^ -f: take PATTERNS from FILE
    , ignoreCase :: Bool
    -- ^ -i: ignore case distinctions in patterns and data
    , noIgnoreCase :: Bool
    -- ^ do not ignore case distinctions (default)
    , wordRegexp :: Bool
    -- ^ -w: match only whole words
    , lineRegexp :: Bool
    -- ^ -x: match only whole lines
    , nullData :: Bool
    -- ^ -z: a data line ends in 0 byte, not newline
    , noMessages :: Bool
    -- ^ -s: suppress error messages
    , invertMatch :: Bool
    -- ^ -v: select non-matching lines
    , maxCount :: Maybe Int
    -- ^ -m: stop after NUM selected lines
    , byteOffset :: Bool
    -- ^ -b: print the byte offset with output lines
    , lineNumber :: Bool
    -- ^ -n: print line number with output lines
    , lineBuffered :: Bool
    -- ^ flush output on every line
    , withFilename :: Bool
    -- ^ -H: print file name with output lines
    , noFilename :: Bool
    -- ^ -h: suppress the file name prefix on output
    , label :: Maybe Text
    -- ^ use LABEL as the standard input file name prefix
    , onlyMatching :: Bool
    -- ^ -o: show only nonempty parts of lines that match
    , quiet :: Bool
    -- ^ -q: , --silent     suppress all normal output
    , binaryFiles :: Maybe Text
    -- ^ assume that binary files are TYPE;
    , text :: Bool
    -- ^ -a: equivalent to --binary-files=text
    , optI :: Bool
    -- ^ -I: equivalent to --binary-files=without-match
    , directories :: Maybe Text
    -- ^ -d: how to handle directories;
    , devices :: Maybe Text
    -- ^ -D: how to handle devices, FIFOs and sockets;
    , recursive :: Bool
    -- ^ -r: like --directories=recurse
    , dereferenceRecursive :: Bool
    -- ^ -R: likewise, but follow all symlinks
    , include :: Maybe Text
    -- ^ search only files that match GLOB (a file pattern)
    , exclude :: Maybe Text
    -- ^ skip files that match GLOB
    , excludeFrom :: Maybe FilePath
    -- ^ skip files that match any file pattern from FILE
    , excludeDir :: Maybe Text
    -- ^ skip directories that match GLOB
    , filesWithoutMatch :: Bool
    -- ^ -L: print only names of FILEs with no selected lines
    , filesWithMatches :: Bool
    -- ^ -l: print only names of FILEs with selected lines
    , count :: Bool
    -- ^ -c: print only a count of selected lines per FILE
    , initialTab :: Bool
    -- ^ -T: make tabs line up (if needed)
    , null_ :: Bool
    -- ^ -Z: print 0 byte after FILE name
    , beforeContext :: Maybe Int
    -- ^ -B: print NUM lines of leading context
    , afterContext :: Maybe Int
    -- ^ -A: print NUM lines of trailing context
    , context :: Maybe Int
    -- ^ -C: print NUM lines of output context
    , optN :: Bool
    -- ^ -N: UM                      same as --context=NUM
    , groupSeparator :: Maybe Text
    -- ^ print SEP on line between matches with context
    , noGroupSeparator :: Bool
    -- ^ do not print separator for matches with context
    , color :: Maybe Text
    -- ^ ,
    , colour :: Maybe Text
    -- ^ use markers to highlight the matching strings;
    , binary :: Bool
    -- ^ -U: do not strip CR characters at EOL (MSDOS/Windows)
    }
    deriving (Show, Eq)

-- | Default options
defaults :: Options
defaults =
    Options
        { extendedRegexp = Nothing
        , fixedStrings = Nothing
        , basicRegexp = Nothing
        , perlRegexp = Nothing
        , regexp = Nothing
        , file = Nothing
        , ignoreCase = False
        , noIgnoreCase = False
        , wordRegexp = False
        , lineRegexp = False
        , nullData = False
        , noMessages = False
        , invertMatch = False
        , maxCount = Nothing
        , byteOffset = False
        , lineNumber = False
        , lineBuffered = False
        , withFilename = False
        , noFilename = False
        , label = Nothing
        , onlyMatching = False
        , quiet = False
        , binaryFiles = Nothing
        , text = False
        , optI = False
        , directories = Nothing
        , devices = Nothing
        , recursive = False
        , dereferenceRecursive = False
        , include = Nothing
        , exclude = Nothing
        , excludeFrom = Nothing
        , excludeDir = Nothing
        , filesWithoutMatch = False
        , filesWithMatches = False
        , count = False
        , initialTab = False
        , null_ = False
        , beforeContext = Nothing
        , afterContext = Nothing
        , context = Nothing
        , optN = False
        , groupSeparator = Nothing
        , noGroupSeparator = False
        , color = Nothing
        , colour = Nothing
        , binary = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ opt extendedRegexp "--extended-regexp"
        , opt fixedStrings "--fixed-strings"
        , opt basicRegexp "--basic-regexp"
        , opt perlRegexp "--perl-regexp"
        , opt regexp "--regexp"
        , optShow file "--file"
        , flag ignoreCase "--ignore-case"
        , flag noIgnoreCase "--no-ignore-case"
        , flag wordRegexp "--word-regexp"
        , flag lineRegexp "--line-regexp"
        , flag nullData "--null-data"
        , flag noMessages "--no-messages"
        , flag invertMatch "--invert-match"
        , optShow maxCount "--max-count"
        , flag byteOffset "--byte-offset"
        , flag lineNumber "--line-number"
        , flag lineBuffered "--line-buffered"
        , flag withFilename "--with-filename"
        , flag noFilename "--no-filename"
        , opt label "--label"
        , flag onlyMatching "--only-matching"
        , flag quiet "--quiet"
        , opt binaryFiles "--binary-files"
        , flag text "--text"
        , flag optI "-I"
        , opt directories "--directories"
        , opt devices "--devices"
        , flag recursive "--recursive"
        , flag dereferenceRecursive "--dereference-recursive"
        , opt include "--include"
        , opt exclude "--exclude"
        , optShow excludeFrom "--exclude-from"
        , opt excludeDir "--exclude-dir"
        , flag filesWithoutMatch "--files-without-match"
        , flag filesWithMatches "--files-with-matches"
        , flag count "--count"
        , flag initialTab "--initial-tab"
        , flag null_ "--null"
        , optShow beforeContext "--before-context"
        , optShow afterContext "--after-context"
        , optShow context "--context"
        , flag optN "-N"
        , opt groupSeparator "--group-separator"
        , flag noGroupSeparator "--no-group-separator"
        , opt color "--color"
        , opt colour "--colour"
        , flag binary "--binary"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

-- | Run grep with options and additional arguments
grep :: Options -> [Text] -> Sh Text
grep opts args = run "grep" (buildArgs opts ++ args)

-- | Run grep, ignoring output
grep_ :: Options -> [Text] -> Sh ()
grep_ opts args = run_ "grep" (buildArgs opts ++ args)
