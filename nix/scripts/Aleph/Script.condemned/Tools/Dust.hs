{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Dust
Description : Typed wrapper for dust

This module was auto-generated from @dust --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Dust (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    dust,
    dust_,
) where

import Aleph.Script hiding (FilePath)
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed:

> defaults { ignoreCase = True, hidden = True }
-}
data Options = Options
    -- \** Options
    { depth :: Maybe Text
    , threads :: Maybe Text
    , config :: Maybe FilePath
    , numberOfLines :: Maybe Text
    , fullPaths :: Bool
    , ignoreDirectory :: Maybe FilePath
    , ignoreAllInFile :: Maybe FilePath
    , dereferenceLinks :: Bool
    , limitFilesystem :: Bool
    , apparentSize :: Bool
    , reverse_ :: Bool
    , noColors :: Bool
    , forceColors :: Bool
    , noPercentBars :: Bool
    , barsOnRight :: Bool
    , minSize :: Maybe Text
    , screenReader :: Bool
    , skipTotal :: Bool
    , filecount :: Bool
    , ignoreHidden :: Bool
    , invertFilter :: Maybe Text
    , filter_ :: Maybe Text
    , fileTypes :: Bool
    , terminalWidth :: Maybe Text
    , noProgress :: Bool
    , printErrors :: Bool
    , onlyDir :: Bool
    , onlyFile :: Bool
    , outputFormat :: Maybe Text
    , stackSize :: Maybe Text
    , outputJson :: Bool
    , mtime :: Maybe Text
    , atime :: Maybe Text
    , ctime :: Maybe Text
    , files0From :: Maybe Text
    , collapse :: Maybe Text
    , filetime :: Maybe Text
    }
    deriving (Show, Eq)

-- | Default options - minimal flags, let the tool use its defaults
defaults :: Options
defaults =
    Options
        { depth = Nothing
        , threads = Nothing
        , config = Nothing
        , numberOfLines = Nothing
        , fullPaths = False
        , ignoreDirectory = Nothing
        , ignoreAllInFile = Nothing
        , dereferenceLinks = False
        , limitFilesystem = False
        , apparentSize = False
        , reverse_ = False
        , noColors = False
        , forceColors = False
        , noPercentBars = False
        , barsOnRight = False
        , minSize = Nothing
        , screenReader = False
        , skipTotal = False
        , filecount = False
        , ignoreHidden = False
        , invertFilter = Nothing
        , filter_ = Nothing
        , fileTypes = False
        , terminalWidth = Nothing
        , noProgress = False
        , printErrors = False
        , onlyDir = False
        , onlyFile = False
        , outputFormat = Nothing
        , stackSize = Nothing
        , outputJson = False
        , mtime = Nothing
        , atime = Nothing
        , ctime = Nothing
        , files0From = Nothing
        , collapse = Nothing
        , filetime = Nothing
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ opt depth "--depth"
        , opt threads "--threads"
        , optShow config "--config"
        , opt numberOfLines "--number-of-lines"
        , flag fullPaths "--full-paths"
        , optShow ignoreDirectory "--ignore-directory"
        , optShow ignoreAllInFile "--ignore-all-in-file"
        , flag dereferenceLinks "--dereference-links"
        , flag limitFilesystem "--limit-filesystem"
        , flag apparentSize "--apparent-size"
        , flag reverse_ "--reverse"
        , flag noColors "--no-colors"
        , flag forceColors "--force-colors"
        , flag noPercentBars "--no-percent-bars"
        , flag barsOnRight "--bars-on-right"
        , opt minSize "--min-size"
        , flag screenReader "--screen-reader"
        , flag skipTotal "--skip-total"
        , flag filecount "--filecount"
        , flag ignoreHidden "--ignore-hidden"
        , opt invertFilter "--invert-filter"
        , opt filter_ "--filter"
        , flag fileTypes "--file-types"
        , opt terminalWidth "--terminal-width"
        , flag noProgress "--no-progress"
        , flag printErrors "--print-errors"
        , flag onlyDir "--only-dir"
        , flag onlyFile "--only-file"
        , opt outputFormat "--output-format"
        , opt stackSize "--stack-size"
        , flag outputJson "--output-json"
        , opt mtime "--mtime"
        , opt atime "--atime"
        , opt ctime "--ctime"
        , opt files0From "--files0-from"
        , opt collapse "--collapse"
        , opt filetime "--filetime"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

{- | Run dust with options and additional arguments

Returns stdout. Throws on non-zero exit.
-}
dust :: Options -> [Text] -> Sh Text
dust opts args = run "dust" (buildArgs opts ++ args)

-- | Run dust, ignoring output
dust_ :: Options -> [Text] -> Sh ()
dust_ opts args = run_ "dust" (buildArgs opts ++ args)
