{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Zoxide
Description : Typed wrapper for zoxide

This module was auto-generated from @zoxide --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Zoxide (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    zoxide,
    zoxide_,
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

{- | Run zoxide with options and additional arguments

Returns stdout. Throws on non-zero exit.
-}
zoxide :: Options -> [Text] -> Sh Text
zoxide opts args = run "zoxide" (buildArgs opts ++ args)

-- | Run zoxide, ignoring output
zoxide_ :: Options -> [Text] -> Sh ()
zoxide_ opts args = run_ "zoxide" (buildArgs opts ++ args)
