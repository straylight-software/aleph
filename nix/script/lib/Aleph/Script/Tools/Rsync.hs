{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Rsync
Description : Typed wrapper for rsync

This module was auto-generated from @rsync --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Rsync (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    rsync,
    rsync_,
) where

import Aleph.Script
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed.
-}
data Options = Options
    { _placeholder :: ()
    }
    deriving (Show, Eq)

-- | Default options
defaults :: Options
defaults =
    Options
        { _placeholder = ()
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{_placeholder = _} =
    catMaybes
        []

-- | Run rsync with options and additional arguments
rsync :: Options -> [Text] -> Sh Text
rsync opts args = run "rsync" (buildArgs opts ++ args)

-- | Run rsync, ignoring output
rsync_ :: Options -> [Text] -> Sh ()
rsync_ opts args = run_ "rsync" (buildArgs opts ++ args)
