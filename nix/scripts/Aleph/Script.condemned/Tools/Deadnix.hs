{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Deadnix
Description : Typed wrapper for deadnix

This module was auto-generated from @deadnix --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Deadnix (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    deadnix,
    deadnix_,
) where

import Aleph.Script
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed:

> defaults { ignoreCase = True, hidden = True }
-}
data Options = Options
    -- \** Options
    { noLambdaArg :: Bool
    -- ^ -l: Don't check lambda parameter arguments
    , noLambdaPatternNames :: Bool
    -- ^ -L: Don't check lambda attrset pattern names (don't break nixpkg
    , quiet :: Bool
    -- ^ -q: Don't print dead code report
    , edit :: Bool
    -- ^ -e: Remove unused code and write to source file
    , hidden :: Bool
    -- ^ -h: Recurse into hidden subdirectories and process hidden .*.nix
    , fail :: Bool
    -- ^ -f: Exit with 1 if unused code has been found
    , outputFormat :: Maybe Text
    -- ^ -o: Output format to use [default: human-readable] [possible val
    , exclude :: Maybe Text
    -- ^ ...          Files to exclude from analysis
    }
    deriving (Show, Eq)

-- | Default options - minimal flags, let the tool use its defaults
defaults :: Options
defaults =
    Options
        { noLambdaArg = False
        , noLambdaPatternNames = False
        , quiet = False
        , edit = False
        , hidden = False
        , fail = False
        , outputFormat = Nothing
        , exclude = Nothing
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag noLambdaArg "--no-lambda-arg"
        , flag noLambdaPatternNames "--no-lambda-pattern-names"
        , flag quiet "--quiet"
        , flag edit "--edit"
        , flag hidden "--hidden"
        , flag fail "--fail"
        , opt outputFormat "--output-format"
        , opt exclude "--exclude"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

{- | Run deadnix with options and additional arguments

Returns stdout. Throws on non-zero exit.
-}
deadnix :: Options -> [Text] -> Sh Text
deadnix opts args = run "deadnix" (buildArgs opts ++ args)

-- | Run deadnix, ignoring output
deadnix_ :: Options -> [Text] -> Sh ()
deadnix_ opts args = run_ "deadnix" (buildArgs opts ++ args)
