{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Tokei
Description : Typed wrapper for tokei

This module was auto-generated from @tokei --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Tokei (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    tokei,
    tokei_,
) where

import Aleph.Script
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed:

> defaults { ignoreCase = True, hidden = True }
-}
data Options = Options
    -- \** Options
    { columns :: Maybe Text
    -- ^ -c: Sets a strict column width of the output, only available for
    , exclude :: Maybe Text
    -- ^ -e: Ignore all files & directories matching the pattern.
    , files :: Bool
    -- ^ -f: Will print out statistics on individual files.
    , input :: Maybe Text
    -- ^ -i: Gives statistics from a previous tokei run. Can be given a
    , hidden :: Bool
    -- ^ Count hidden files.
    , languages :: Bool
    -- ^ -l: Prints out supported languages and their extensions.
    , noIgnore :: Bool
    -- ^ Don't respect ignore files (.gitignore, .ignore, etc.). This
    , noIgnoreParent :: Bool
    -- ^ Don't respect ignore files (.gitignore, .ignore, etc.) in
    , noIgnoreDot :: Bool
    -- ^ Don't respect .ignore and .tokeignore files, including those
    , noIgnoreVcs :: Bool
    -- ^ Don't respect VCS ignore files (.gitignore, .hgignore, etc.)
    , output :: Maybe Text
    -- ^ -o: Outputs Tokei in a specific format. Compile with additional
    , streaming :: Maybe Text
    -- ^ prints the (language, path, lines, blanks, code, comments)
    , sort :: Maybe Text
    -- ^ -s: Sort languages based on column [possible values: files,
    , rsort :: Maybe Text
    -- ^ -r: Reverse sort languages based on column [possible values:
    , types :: Maybe Text
    -- ^ -t: Filters output by language type, separated by a comma. i.e.
    , compact :: Bool
    -- ^ -C: Do not print statistics about embedded languages.
    , numFormat :: Maybe Text
    -- ^ -n: Format of printed numbers, i.e., plain (1234, default),
    , verbose :: Bool
    -- ^ -v: ...                     Set log output level:
    }
    deriving (Show, Eq)

-- | Default options - minimal flags, let the tool use its defaults
defaults :: Options
defaults =
    Options
        { columns = Nothing
        , exclude = Nothing
        , files = False
        , input = Nothing
        , hidden = False
        , languages = False
        , noIgnore = False
        , noIgnoreParent = False
        , noIgnoreDot = False
        , noIgnoreVcs = False
        , output = Nothing
        , streaming = Nothing
        , sort = Nothing
        , rsort = Nothing
        , types = Nothing
        , compact = False
        , numFormat = Nothing
        , verbose = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ opt columns "--columns"
        , opt exclude "--exclude"
        , flag files "--files"
        , opt input "--input"
        , flag hidden "--hidden"
        , flag languages "--languages"
        , flag noIgnore "--no-ignore"
        , flag noIgnoreParent "--no-ignore-parent"
        , flag noIgnoreDot "--no-ignore-dot"
        , flag noIgnoreVcs "--no-ignore-vcs"
        , opt output "--output"
        , opt streaming "--streaming"
        , opt sort "--sort"
        , opt rsort "--rsort"
        , opt types "--types"
        , flag compact "--compact"
        , opt numFormat "--num-format"
        , flag verbose "--verbose"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

{- | Run tokei with options and additional arguments

Returns stdout. Throws on non-zero exit.
-}
tokei :: Options -> [Text] -> Sh Text
tokei opts args = run "tokei" (buildArgs opts ++ args)

-- | Run tokei, ignoring output
tokei_ :: Options -> [Text] -> Sh ()
tokei_ opts args = run_ "tokei" (buildArgs opts ++ args)
