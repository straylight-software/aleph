{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Xargs
Description : Typed wrapper for xargs

This module was auto-generated from @xargs --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Xargs (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    xargs,
    xargs_,
) where

import Aleph.Script hiding (FilePath)
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed.
-}
data Options = Options
    { null_ :: Bool
    -- ^ -0: items are separated by a null, not whitespace;
    , argFile :: Maybe FilePath
    -- ^ -a: read arguments from FILE, not standard input
    , delimiter :: Maybe Text
    -- ^ -d: items in input stream are separated by CHARACTER,
    , optE :: Maybe Text
    -- ^ -E: set logical EOF string; if END occurs as a line
    , eof :: Maybe Text
    -- ^ -e: equivalent to -E END if END is specified;
    , optI :: Maybe Text
    -- ^ -I: same as --replace=R
    , replace :: Maybe Text
    -- ^ -i: replace R in INITIAL-ARGS with names read
    , maxLines :: Maybe Text
    -- ^ -L: use at most MAX-LINES non-blank input lines per
    , optL :: Maybe Text
    -- ^ -l: similar to -L but defaults to at most one non-
    , maxArgs :: Maybe Text
    -- ^ -n: use at most MAX-ARGS arguments per command line
    , openTty :: Maybe Text
    -- ^ -o: eopen stdin as /dev/tty in the child process
    , maxProcs :: Maybe Text
    -- ^ -P: run at most MAX-PROCS processes at a time
    , interactive :: Bool
    -- ^ -p: prompt before running commands
    , processSlotVar :: Maybe Text
    -- ^ set environment variable VAR in child processes
    , noRunIfEmpty :: Bool
    -- ^ -r: if there are no arguments, then do not run COMMAND
    , maxChars :: Maybe Text
    -- ^ -s: limit length of command line to MAX-CHARS
    , showLimits :: Bool
    -- ^ show limits on command-line length
    , verbose :: Bool
    -- ^ -t: print commands before executing them
    , exit :: Bool
    -- ^ -x: exit if the size (see -s) is exceeded
    }
    deriving (Show, Eq)

-- | Default options
defaults :: Options
defaults =
    Options
        { null_ = False
        , argFile = Nothing
        , delimiter = Nothing
        , optE = Nothing
        , eof = Nothing
        , optI = Nothing
        , replace = Nothing
        , maxLines = Nothing
        , optL = Nothing
        , maxArgs = Nothing
        , openTty = Nothing
        , maxProcs = Nothing
        , interactive = False
        , processSlotVar = Nothing
        , noRunIfEmpty = False
        , maxChars = Nothing
        , showLimits = False
        , verbose = False
        , exit = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag null_ "--null"
        , optShow argFile "--arg-file"
        , opt delimiter "--delimiter"
        , opt optE "-E"
        , opt eof "--eof"
        , opt optI "-I"
        , opt replace "--replace"
        , opt maxLines "--max-lines"
        , opt optL "-l"
        , opt maxArgs "--max-args"
        , opt openTty "--open-tty"
        , opt maxProcs "--max-procs"
        , flag interactive "--interactive"
        , opt processSlotVar "--process-slot-var"
        , flag noRunIfEmpty "--no-run-if-empty"
        , opt maxChars "--max-chars"
        , flag showLimits "--show-limits"
        , flag verbose "--verbose"
        , flag exit "--exit"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

-- | Run xargs with options and additional arguments
xargs :: Options -> [Text] -> Sh Text
xargs opts args = run "xargs" (buildArgs opts ++ args)

-- | Run xargs, ignoring output
xargs_ :: Options -> [Text] -> Sh ()
xargs_ opts args = run_ "xargs" (buildArgs opts ++ args)
