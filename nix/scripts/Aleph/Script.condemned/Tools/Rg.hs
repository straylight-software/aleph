{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Rg
Description : Typed wrapper for ripgrep

Ripgrep is a line-oriented search tool that recursively searches
the current directory for a regex pattern.

@
import Aleph.Script
import qualified Aleph.Script.Tools.Rg as Rg

main = script $ do
  -- Find all TODOs in Haskell files
  output <- Rg.rg Rg.defaults
    { Rg.ignoreCase = True
    , Rg.glob = Just "*.hs"
    } "TODO" ["."]
  echo output
@
-}
module Aleph.Script.Tools.Rg (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    rg,
    rg_,

    -- * Common patterns
    search,
    searchFiles,
    countFiles,
    listFiles,
) where

import Aleph.Script hiding (FilePath)
import Data.Maybe (catMaybes)
import qualified Data.Text as T
import Prelude (Bool (..), Eq, FilePath, Int, Maybe (..), Show, filter, map, not, return, show, ($), (++), (.), (<>))

{- | Ripgrep options

Use 'defaults' and override fields as needed:

> defaults { ignoreCase = True, hidden = True }
-}
data Options = Options
    { -- \** Pattern matching
      ignoreCase :: Bool
    -- ^ -i: Case insensitive search
    , smartCase :: Bool
    -- ^ -S: Smart case (case-insensitive if pattern is lowercase)
    , fixedStrings :: Bool
    -- ^ -F: Treat pattern as literal string, not regex
    , wordRegexp :: Bool
    -- ^ -w: Match whole words only
    , lineRegexp :: Bool
    -- ^ -x: Match whole lines only
    , multiline :: Bool
    -- ^ -U: Enable multiline matching
    , pcre2 :: Bool
    -- ^ -P: Use PCRE2 regex engine
    , -- \** File filtering
      glob :: Maybe Text
    -- ^ -g: Include/exclude files matching glob
    , iglob :: Maybe Text
    -- ^ --iglob: Case-insensitive glob
    , type_ :: Maybe Text
    -- ^ -t: Only search files of TYPE (e.g., "hs", "py")
    , typeNot :: Maybe Text
    -- ^ -T: Exclude files of TYPE
    , hidden :: Bool
    -- ^ --hidden: Search hidden files/directories
    , noIgnore :: Bool
    -- ^ --no-ignore: Don't use .gitignore
    , follow :: Bool
    -- ^ -L: Follow symlinks
    , maxDepth :: Maybe Int
    -- ^ --max-depth: Max directory depth
    , maxFilesize :: Maybe Text
    -- ^ --max-filesize: Skip files larger than NUM
    , -- \** Output control
      lineNumber :: Bool
    -- ^ -n: Show line numbers (default in TTY)
    , noLineNumber :: Bool
    -- ^ -N: Suppress line numbers
    , column :: Bool
    -- ^ --column: Show column numbers
    , onlyMatching :: Bool
    -- ^ -o: Print only matched parts
    , count :: Bool
    -- ^ -c: Show count per file
    , countMatches :: Bool
    -- ^ --count-matches: Show total match count per file
    , filesWithMatches :: Bool
    -- ^ -l: Only print filenames with matches
    , filesWithoutMatch :: Bool
    -- ^ --files-without-match: Print files without matches
    , context :: Maybe Int
    -- ^ -C: Show NUM lines before and after
    , beforeContext :: Maybe Int
    -- ^ -B: Show NUM lines before match
    , afterContext :: Maybe Int
    -- ^ -A: Show NUM lines after match
    , color :: Maybe Text
    -- ^ --color: When to use color (never/auto/always)
    , heading :: Bool
    -- ^ --heading: Group matches by file (default in TTY)
    , noHeading :: Bool
    -- ^ --no-heading: Don't group by file
    , json :: Bool
    -- ^ --json: Output in JSON Lines format
    , vimgrep :: Bool
    -- ^ --vimgrep: Output in vim-compatible format
    , -- \** Search behavior
      invertMatch :: Bool
    -- ^ -v: Invert match (show non-matching lines)
    , maxCount :: Maybe Int
    -- ^ -m: Stop after NUM matches per file
    , threads :: Maybe Int
    -- ^ -j: Number of threads
    , quiet :: Bool
    -- ^ -q: Suppress output, exit 0 if match found
    , replace :: Maybe Text
    -- ^ -r: Replace matches with TEXT
    , passthru :: Bool
    -- ^ --passthru: Print all lines, highlight matches
    }
    deriving (Show, Eq)

-- | Default options - minimal flags, let ripgrep use its defaults
defaults :: Options
defaults =
    Options
        { ignoreCase = False
        , smartCase = False
        , fixedStrings = False
        , wordRegexp = False
        , lineRegexp = False
        , multiline = False
        , pcre2 = False
        , glob = Nothing
        , iglob = Nothing
        , type_ = Nothing
        , typeNot = Nothing
        , hidden = False
        , noIgnore = False
        , follow = False
        , maxDepth = Nothing
        , maxFilesize = Nothing
        , lineNumber = False
        , noLineNumber = False
        , column = False
        , onlyMatching = False
        , count = False
        , countMatches = False
        , filesWithMatches = False
        , filesWithoutMatch = False
        , context = Nothing
        , beforeContext = Nothing
        , afterContext = Nothing
        , color = Nothing
        , heading = False
        , noHeading = False
        , json = False
        , vimgrep = False
        , invertMatch = False
        , maxCount = Nothing
        , threads = Nothing
        , quiet = False
        , replace = Nothing
        , passthru = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag ignoreCase "-i"
        , flag smartCase "-S"
        , flag fixedStrings "-F"
        , flag wordRegexp "-w"
        , flag lineRegexp "-x"
        , flag multiline "-U"
        , flag pcre2 "-P"
        , opt glob "-g"
        , opt iglob "--iglob"
        , opt type_ "-t"
        , opt typeNot "-T"
        , flag hidden "--hidden"
        , flag noIgnore "--no-ignore"
        , flag follow "-L"
        , optShow maxDepth "--max-depth"
        , opt maxFilesize "--max-filesize"
        , flag lineNumber "-n"
        , flag noLineNumber "-N"
        , flag column "--column"
        , flag onlyMatching "-o"
        , flag count "-c"
        , flag countMatches "--count-matches"
        , flag filesWithMatches "-l"
        , flag filesWithoutMatch "--files-without-match"
        , optShow context "-C"
        , optShow beforeContext "-B"
        , optShow afterContext "-A"
        , opt color "--color"
        , flag heading "--heading"
        , flag noHeading "--no-heading"
        , flag json "--json"
        , flag vimgrep "--vimgrep"
        , flag invertMatch "-v"
        , optShow maxCount "-m"
        , optShow threads "-j"
        , flag quiet "-q"
        , opt replace "-r"
        , flag passthru "--passthru"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

{- | Run ripgrep with options, pattern, and paths

Returns stdout. Throws on non-zero exit (unless using 'errExit False').
-}
rg :: Options -> Text -> [FilePath] -> Sh Text
rg opts pat paths =
    run "rg" (buildArgs opts ++ [pat] ++ map pack paths)

-- | Run ripgrep, ignoring output
rg_ :: Options -> Text -> [FilePath] -> Sh ()
rg_ opts pat paths =
    run_ "rg" (buildArgs opts ++ [pat] ++ map pack paths)

-- | Simple search: pattern in paths with default options
search :: Text -> [FilePath] -> Sh Text
search = rg defaults

-- | Search returning just the matching filenames
searchFiles :: Text -> [FilePath] -> Sh [FilePath]
searchFiles pat paths = do
    output <- rg defaults{filesWithMatches = True} pat paths
    return $ map unpack $ filter (not . T.null) $ T.lines output

-- | Count matches per file
countFiles :: Text -> [FilePath] -> Sh Text
countFiles pat paths =
    rg defaults{count = True} pat paths

-- | List files that would be searched (no pattern matching)
listFiles :: [FilePath] -> Sh [FilePath]
listFiles paths = do
    output <- run "rg" ("--files" : map pack paths)
    return $ map unpack $ filter (not . T.null) $ T.lines output
