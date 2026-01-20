{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Statix
Description : Typed wrapper for statix

This module was auto-generated from @statix --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Statix (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    statix,
    statix_,
) where

import Aleph.Script
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed:

> defaults { ignoreCase = True, hidden = True }
-}
data Options = Options
    { _placeholder :: ()
    -- ^ No options found
    }
    deriving (Show, Eq)

-- | Default options - minimal flags, let the tool use its defaults
defaults :: Options
defaults =
    Options
        { _placeholder = ()
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        []
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

{- | Run statix with options and additional arguments

Returns stdout. Throws on non-zero exit.
-}
statix :: Options -> [Text] -> Sh Text
statix opts args = run "statix" (buildArgs opts ++ args)

-- | Run statix, ignoring output
statix_ :: Options -> [Text] -> Sh ()
statix_ opts args = run_ "statix" (buildArgs opts ++ args)
