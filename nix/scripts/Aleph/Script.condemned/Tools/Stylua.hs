{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Stylua
Description : Typed wrapper for stylua

This module was auto-generated from @stylua --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Stylua (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    stylua,
    stylua_,
) where

import Aleph.Script
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed:

> defaults { ignoreCase = True, hidden = True }
-}
data Options = Options
    -- \** OPTIONS
    { allowHidden :: Bool
    , check :: Bool
    , configPath :: Maybe Text
    , glob :: Maybe Text
    , searchParentDirectories :: Bool
    , verbose :: Bool
    }
    deriving (Show, Eq)

-- | Default options - minimal flags, let the tool use its defaults
defaults :: Options
defaults =
    Options
        { allowHidden = False
        , check = False
        , configPath = Nothing
        , glob = Nothing
        , searchParentDirectories = False
        , verbose = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag allowHidden "--allow-hidden"
        , flag check "--check"
        , opt configPath "--config-path"
        , opt glob "--glob"
        , flag searchParentDirectories "--search-parent-directories"
        , flag verbose "--verbose"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

{- | Run stylua with options and additional arguments

Returns stdout. Throws on non-zero exit.
-}
stylua :: Options -> [Text] -> Sh Text
stylua opts args = run "stylua" (buildArgs opts ++ args)

-- | Run stylua, ignoring output
stylua_ :: Options -> [Text] -> Sh ()
stylua_ opts args = run_ "stylua" (buildArgs opts ++ args)
