{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Hyperfine
Description : Typed wrapper for hyperfine

This module was auto-generated from @hyperfine --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Hyperfine (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    hyperfine,
    hyperfine_,
) where

import Aleph.Script hiding (FilePath)
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed:

> defaults { ignoreCase = True, hidden = True }
-}
data Options = Options
    -- \** Options
    { warmup :: Maybe Int
    , minRuns :: Maybe Int
    , maxRuns :: Maybe Int
    , runs :: Maybe Int
    , setup :: Maybe Text
    , reference :: Maybe Text
    , prepare :: Maybe Text
    , conclude :: Maybe Text
    , cleanup :: Maybe Text
    , parameterScan :: Maybe Text
    -- ^ -P: <MIN> <MAX>
    , parameterStepSize :: Maybe Text
    , parameterList :: Maybe Text
    -- ^ -L: <VALUES>
    , shell :: Maybe Text
    , optN :: Bool
    , ignoreFailure :: Bool
    , style :: Maybe Text
    , sort :: Maybe Text
    , timeUnit :: Maybe Text
    , exportAsciidoc :: Maybe FilePath
    , exportCsv :: Maybe FilePath
    , exportJson :: Maybe FilePath
    , exportMarkdown :: Maybe FilePath
    , exportOrgmode :: Maybe FilePath
    , showOutput :: Bool
    , output :: Maybe Text
    , input :: Maybe Text
    , commandName :: Maybe Text
    }
    deriving (Show, Eq)

-- | Default options - minimal flags, let the tool use its defaults
defaults :: Options
defaults =
    Options
        { warmup = Nothing
        , minRuns = Nothing
        , maxRuns = Nothing
        , runs = Nothing
        , setup = Nothing
        , reference = Nothing
        , prepare = Nothing
        , conclude = Nothing
        , cleanup = Nothing
        , parameterScan = Nothing
        , parameterStepSize = Nothing
        , parameterList = Nothing
        , shell = Nothing
        , optN = False
        , ignoreFailure = False
        , style = Nothing
        , sort = Nothing
        , timeUnit = Nothing
        , exportAsciidoc = Nothing
        , exportCsv = Nothing
        , exportJson = Nothing
        , exportMarkdown = Nothing
        , exportOrgmode = Nothing
        , showOutput = False
        , output = Nothing
        , input = Nothing
        , commandName = Nothing
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ optShow warmup "--warmup"
        , optShow minRuns "--min-runs"
        , optShow maxRuns "--max-runs"
        , optShow runs "--runs"
        , opt setup "--setup"
        , opt reference "--reference"
        , opt prepare "--prepare"
        , opt conclude "--conclude"
        , opt cleanup "--cleanup"
        , opt parameterScan "--parameter-scan"
        , opt parameterStepSize "--parameter-step-size"
        , opt parameterList "--parameter-list"
        , opt shell "--shell"
        , flag optN "-N"
        , flag ignoreFailure "--ignore-failure"
        , opt style "--style"
        , opt sort "--sort"
        , opt timeUnit "--time-unit"
        , optShow exportAsciidoc "--export-asciidoc"
        , optShow exportCsv "--export-csv"
        , optShow exportJson "--export-json"
        , optShow exportMarkdown "--export-markdown"
        , optShow exportOrgmode "--export-orgmode"
        , flag showOutput "--show-output"
        , opt output "--output"
        , opt input "--input"
        , opt commandName "--command-name"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

{- | Run hyperfine with options and additional arguments

Returns stdout. Throws on non-zero exit.
-}
hyperfine :: Options -> [Text] -> Sh Text
hyperfine opts args = run "hyperfine" (buildArgs opts ++ args)

-- | Run hyperfine, ignoring output
hyperfine_ :: Options -> [Text] -> Sh ()
hyperfine_ opts args = run_ "hyperfine" (buildArgs opts ++ args)
