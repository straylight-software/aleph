{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Bat
Description : Typed wrapper for bat

This module was auto-generated from @bat --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Bat (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    bat,
    bat_,
) where

import Aleph.Script
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed:

> defaults { ignoreCase = True, hidden = True }
-}
data Options = Options
    -- \** Options
    { showAll :: Bool
    , nonprintableNotation :: Maybe Text
    , binary :: Maybe Text
    , plain :: Bool
    -- ^ -p: ...
    , language :: Maybe Text
    , highlightLine :: Maybe Text
    , fileName :: Maybe Text
    , diff :: Bool
    , diffContext :: Maybe Int
    , tabs :: Maybe Text
    , wrap :: Maybe Text
    , chopLongLines :: Bool
    , terminalWidth :: Maybe Text
    , number :: Bool
    , color :: Maybe Text
    , italicText :: Maybe Text
    , decorations :: Maybe Text
    , forceColorization :: Bool
    , paging :: Maybe Text
    , pager :: Maybe Text
    , mapSyntax :: Maybe Text
    , ignoredSuffix :: Maybe Text
    , theme :: Maybe Text
    , themeLight :: Maybe Text
    , themeDark :: Maybe Text
    , listThemes :: Bool
    , squeezeBlank :: Bool
    , squeezeLimit :: Maybe Text
    , stripAnsi :: Maybe Text
    , style :: Maybe Text
    , lineRange :: Maybe Text
    , listLanguages :: Bool
    , unbuffered :: Bool
    , completion :: Maybe Text
    , diagnostic :: Bool
    , acknowledgements :: Bool
    , setTerminalTitle :: Bool
    }
    deriving (Show, Eq)

-- | Default options - minimal flags, let the tool use its defaults
defaults :: Options
defaults =
    Options
        { showAll = False
        , nonprintableNotation = Nothing
        , binary = Nothing
        , plain = False
        , language = Nothing
        , highlightLine = Nothing
        , fileName = Nothing
        , diff = False
        , diffContext = Nothing
        , tabs = Nothing
        , wrap = Nothing
        , chopLongLines = False
        , terminalWidth = Nothing
        , number = False
        , color = Nothing
        , italicText = Nothing
        , decorations = Nothing
        , forceColorization = False
        , paging = Nothing
        , pager = Nothing
        , mapSyntax = Nothing
        , ignoredSuffix = Nothing
        , theme = Nothing
        , themeLight = Nothing
        , themeDark = Nothing
        , listThemes = False
        , squeezeBlank = False
        , squeezeLimit = Nothing
        , stripAnsi = Nothing
        , style = Nothing
        , lineRange = Nothing
        , listLanguages = False
        , unbuffered = False
        , completion = Nothing
        , diagnostic = False
        , acknowledgements = False
        , setTerminalTitle = False
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ flag showAll "--show-all"
        , opt nonprintableNotation "--nonprintable-notation"
        , opt binary "--binary"
        , flag plain "--plain"
        , opt language "--language"
        , opt highlightLine "--highlight-line"
        , opt fileName "--file-name"
        , flag diff "--diff"
        , optShow diffContext "--diff-context"
        , opt tabs "--tabs"
        , opt wrap "--wrap"
        , flag chopLongLines "--chop-long-lines"
        , opt terminalWidth "--terminal-width"
        , flag number "--number"
        , opt color "--color"
        , opt italicText "--italic-text"
        , opt decorations "--decorations"
        , flag forceColorization "--force-colorization"
        , opt paging "--paging"
        , opt pager "--pager"
        , opt mapSyntax "--map-syntax"
        , opt ignoredSuffix "--ignored-suffix"
        , opt theme "--theme"
        , opt themeLight "--theme-light"
        , opt themeDark "--theme-dark"
        , flag listThemes "--list-themes"
        , flag squeezeBlank "--squeeze-blank"
        , opt squeezeLimit "--squeeze-limit"
        , opt stripAnsi "--strip-ansi"
        , opt style "--style"
        , opt lineRange "--line-range"
        , flag listLanguages "--list-languages"
        , flag unbuffered "--unbuffered"
        , opt completion "--completion"
        , flag diagnostic "--diagnostic"
        , flag acknowledgements "--acknowledgements"
        , flag setTerminalTitle "--set-terminal-title"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

{- | Run bat with options and additional arguments

Returns stdout. Throws on non-zero exit.
-}
bat :: Options -> [Text] -> Sh Text
bat opts args = run "bat" (buildArgs opts ++ args)

-- | Run bat, ignoring output
bat_ :: Options -> [Text] -> Sh ()
bat_ opts args = run_ "bat" (buildArgs opts ++ args)
