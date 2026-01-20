{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Find
Description : Typed wrapper for find

This module was auto-generated from @find --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Find (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    find,
    find_,
) where

import Aleph.Script hiding (find)
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed.
-}
data Options = Options
    { optD :: Bool
    -- ^ -d: aystart -follow -nowarn -regextype -warn
    , optM :: Bool
    -- ^ -m: ount -noleaf -xdev -ignore_readdir_race -noignore_
    , optA :: Bool
    -- ^ -a: min N -anewer FILE -atime N -cmin N -cnewer FILE -
    , optC :: Bool
    -- ^ -c: time N -empty -false -fstype TYPE -gid N -group NA
    , optI :: Bool
    -- ^ -i: name PATTERN -inum N -iwholename PATTERN -iregex P
    , optL :: Bool
    -- ^ -l: inks N -lname PATTERN -mmin N -mtime N -name PATTE
    , optN :: Bool
    -- ^ -n: ouser -nogroup -path PATTERN -perm [-/]MODE -regex
    , optR :: Bool
    -- ^ -r: eadable -writable -executable
    , optW :: Bool
    -- ^ -w: holename PATTERN -size N[bcwkMG] -true -type [bcdp
    , optU :: Bool
    -- ^ -u: sed N -user NAME -xtype [bcdpfls]
    , optF :: Bool
    -- ^ -f: print0 FILE -fprint FILE -ls -fls FILE -prune -qui
    , optE :: Bool
    -- ^ -e: xec COMMAND ; -exec COMMAND {} + -ok COMMAND ;
    }
    deriving (Show, Eq)

-- | Default options
defaults :: Options
defaults =
    Options
        { optD = False
        , optM = False
        , optA = False
        , optC = False
        , optI = False
        , optL = False
        , optN = False
        , optR = False
        , optW = False
        , optU = False
        , optF = False
        , optE = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag optD "-d"
        , flag optM "-m"
        , flag optA "-a"
        , flag optC "-c"
        , flag optI "-i"
        , flag optL "-l"
        , flag optN "-n"
        , flag optR "-r"
        , flag optW "-w"
        , flag optU "-u"
        , flag optF "-f"
        , flag optE "-e"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

-- | Run find with options and additional arguments
find :: Options -> [Text] -> Sh Text
find opts args = run "find" (buildArgs opts ++ args)

-- | Run find, ignoring output
find_ :: Options -> [Text] -> Sh ()
find_ opts args = run_ "find" (buildArgs opts ++ args)
