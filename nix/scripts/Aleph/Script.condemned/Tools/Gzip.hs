{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Gzip
Description : Typed wrapper for gzip

This module was auto-generated from @gzip --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Gzip (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    gzip,
    gzip_,
) where

import Aleph.Script
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed.
-}
data Options = Options
    { stdout :: Bool
    -- ^ -c: write on standard output, keep original files unch
    , decompress :: Bool
    -- ^ -d: decompress
    , force :: Bool
    -- ^ -f: force overwrite of output file and compress links
    , keep :: Bool
    -- ^ -k: keep (don't delete) input files
    , list :: Bool
    -- ^ -l: list compressed file contents
    , license :: Bool
    -- ^ -L: display software license
    , noName :: Bool
    -- ^ -n: do not save or restore the original name and times
    , name :: Bool
    -- ^ -N: save or restore the original name and timestamp
    , quiet :: Bool
    -- ^ -q: suppress all warnings
    , recursive :: Bool
    -- ^ -r: operate recursively on directories
    , rsyncable :: Bool
    -- ^ make rsync-friendly archive
    , suffix :: Maybe Text
    -- ^ -S: use suffix SUF on compressed files
    , synchronous :: Bool
    -- ^ synchronous output (safer if system crashes, but s
    , test :: Bool
    -- ^ -t: test compressed file integrity
    , verbose :: Bool
    -- ^ -v: verbose mode
    , fast :: Bool
    -- ^ -1: compress faster
    , best :: Bool
    -- ^ -9: compress better
    }
    deriving (Show, Eq)

-- | Default options
defaults :: Options
defaults =
    Options
        { stdout = False
        , decompress = False
        , force = False
        , keep = False
        , list = False
        , license = False
        , noName = False
        , name = False
        , quiet = False
        , recursive = False
        , rsyncable = False
        , suffix = Nothing
        , synchronous = False
        , test = False
        , verbose = False
        , fast = False
        , best = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag stdout "--stdout"
        , flag decompress "--decompress"
        , flag force "--force"
        , flag keep "--keep"
        , flag list "--list"
        , flag license "--license"
        , flag noName "--no-name"
        , flag name "--name"
        , flag quiet "--quiet"
        , flag recursive "--recursive"
        , flag rsyncable "--rsyncable"
        , opt suffix "--suffix"
        , flag synchronous "--synchronous"
        , flag test "--test"
        , flag verbose "--verbose"
        , flag fast "--fast"
        , flag best "--best"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

-- | Run gzip with options and additional arguments
gzip :: Options -> [Text] -> Sh Text
gzip opts args = run "gzip" (buildArgs opts ++ args)

-- | Run gzip, ignoring output
gzip_ :: Options -> [Text] -> Sh ()
gzip_ opts args = run_ "gzip" (buildArgs opts ++ args)
