{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Sed
Description : Typed wrapper for sed

This module was auto-generated from @sed --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Sed (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    sed,
    sed_,
) where

import Aleph.Script
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed.
-}
data Options = Options
    { quiet :: Bool
    -- ^ -n: , --silent
    , debug :: Bool
    , optE :: Bool
    -- ^ -e: script, --expression=script
    , optF :: Bool
    -- ^ -f: script-file, --file=script-file
    , followSymlinks :: Bool
    , optI :: Maybe Text
    -- ^ -i: , --in-place[=SUFFIX]
    , optL :: Maybe Int
    -- ^ -l: , --line-length=N
    , posix :: Bool
    , separate :: Bool
    , sandbox :: Bool
    , unbuffered :: Bool
    , nullData :: Bool
    }
    deriving (Show, Eq)

-- | Default options
defaults :: Options
defaults =
    Options
        { quiet = False
        , debug = False
        , optE = False
        , optF = False
        , followSymlinks = False
        , optI = Nothing
        , optL = Nothing
        , posix = False
        , separate = False
        , sandbox = False
        , unbuffered = False
        , nullData = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag quiet "--quiet"
        , flag debug "--debug"
        , flag optE "-e"
        , flag optF "-f"
        , flag followSymlinks "--follow-symlinks"
        , opt optI "-i"
        , optShow optL "-l"
        , flag posix "--posix"
        , flag separate "--separate"
        , flag sandbox "--sandbox"
        , flag unbuffered "--unbuffered"
        , flag nullData "--null-data"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

-- | Run sed with options and additional arguments
sed :: Options -> [Text] -> Sh Text
sed opts args = run "sed" (buildArgs opts ++ args)

-- | Run sed, ignoring output
sed_ :: Options -> [Text] -> Sh ()
sed_ opts args = run_ "sed" (buildArgs opts ++ args)
