{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Taplo
Description : Typed wrapper for taplo

This module was auto-generated from @taplo --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Taplo (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    taplo,
    taplo_,
) where

import Aleph.Script
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed:

> defaults { ignoreCase = True, hidden = True }
-}
data Options = Options
    -- \** Options
    { colors :: Maybe Text
    , verbose :: Bool
    , logSpans :: Bool
    }
    deriving (Show, Eq)

-- | Default options - minimal flags, let the tool use its defaults
defaults :: Options
defaults =
    Options
        { colors = Nothing
        , verbose = False
        , logSpans = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ opt colors "--colors"
        , flag verbose "--verbose"
        , flag logSpans "--log-spans"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

{- | Run taplo with options and additional arguments

Returns stdout. Throws on non-zero exit.
-}
taplo :: Options -> [Text] -> Sh Text
taplo opts args = run "taplo" (buildArgs opts ++ args)

-- | Run taplo, ignoring output
taplo_ :: Options -> [Text] -> Sh ()
taplo_ opts args = run_ "taplo" (buildArgs opts ++ args)
