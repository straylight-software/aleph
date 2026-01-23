{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Delta
Description : Typed wrapper for delta

This module was auto-generated from @delta --help@ output.
Review and adjust field names and types as needed.
-}
module Aleph.Script.Tools.Delta (
    -- * Options
    Options (..),
    defaults,

    -- * Invocation
    delta,
    delta_,
) where

import Aleph.Script hiding (FilePath)
import Data.Maybe (catMaybes)

{- | Options record

Use 'defaults' and override fields as needed:

> defaults { ignoreCase = True, hidden = True }
-}
data Options = Options
    -- \** Options
    { blameCodeStyle :: Maybe Text
    , blameFormat :: Maybe Text
    , blamePalette :: Maybe Text
    , blameSeparatorFormat :: Maybe Text
    , blameSeparatorStyle :: Maybe Text
    , blameTimestampFormat :: Maybe Text
    , blameTimestampOutputFormat :: Maybe Text
    , colorOnly :: Bool
    , config :: Maybe FilePath
    , commitDecorationStyle :: Maybe Text
    , commitRegex :: Maybe Text
    , commitStyle :: Maybe Text
    , dark :: Bool
    , defaultLanguage :: Maybe Text
    , detectDarkLight :: Maybe Text
    , diffHighlight :: Bool
    , diffSoFancy :: Bool
    , diffStatAlignWidth :: Maybe Int
    , features :: Maybe Text
    , fileAddedLabel :: Maybe Text
    , fileCopiedLabel :: Maybe Text
    , fileDecorationStyle :: Maybe Text
    , fileModifiedLabel :: Maybe Text
    , fileRemovedLabel :: Maybe Text
    , fileRenamedLabel :: Maybe Text
    , fileStyle :: Maybe Text
    , fileTransformation :: Maybe Text
    , generateCompletion :: Maybe Text
    , grepContextLineStyle :: Maybe Text
    , grepFileStyle :: Maybe Text
    , grepHeaderDecorationStyle :: Maybe Text
    , grepHeaderFileStyle :: Maybe Text
    , grepLineNumberStyle :: Maybe Text
    , grepOutputType :: Maybe Text
    , grepMatchLineStyle :: Maybe Text
    , grepMatchWordStyle :: Maybe Text
    , grepSeparatorSymbol :: Maybe Text
    , hunkHeaderDecorationStyle :: Maybe Text
    , hunkHeaderFileStyle :: Maybe Text
    , hunkHeaderLineNumberStyle :: Maybe Text
    , hunkHeaderStyle :: Maybe Text
    , hunkLabel :: Maybe Text
    , hyperlinks :: Bool
    , hyperlinksCommitLinkFormat :: Maybe Text
    , hyperlinksFileLinkFormat :: Maybe Text
    , inlineHintStyle :: Maybe Text
    , inspectRawLines :: Maybe Text
    , keepPlusMinusMarkers :: Bool
    , light :: Bool
    , lineBufferSize :: Maybe Int
    , lineFillMethod :: Maybe Text
    , lineNumbers :: Bool
    , lineNumbersLeftFormat :: Maybe Text
    , lineNumbersLeftStyle :: Maybe Text
    , lineNumbersMinusStyle :: Maybe Text
    , lineNumbersPlusStyle :: Maybe Text
    , lineNumbersRightFormat :: Maybe Text
    , lineNumbersRightStyle :: Maybe Text
    , lineNumbersZeroStyle :: Maybe Text
    , listLanguages :: Bool
    , listSyntaxThemes :: Bool
    , mapStyles :: Maybe Text
    , maxLineDistance :: Maybe Text
    , maxSyntaxHighlightingLength :: Maybe Int
    , maxLineLength :: Maybe Int
    , mergeConflictBeginSymbol :: Maybe Text
    , mergeConflictEndSymbol :: Maybe Text
    , mergeConflictOursDiffHeaderDecorationStyle :: Maybe Text
    , mergeConflictOursDiffHeaderStyle :: Maybe Text
    , mergeConflictTheirsDiffHeaderDecorationStyle :: Maybe Text
    , mergeConflictTheirsDiffHeaderStyle :: Maybe Text
    , minusEmptyLineMarkerStyle :: Maybe Text
    , minusEmphStyle :: Maybe Text
    , minusNonEmphStyle :: Maybe Text
    , minusStyle :: Maybe Text
    , navigate :: Bool
    , navigateRegex :: Maybe Text
    , noGitconfig :: Bool
    , pager :: Maybe Text
    , paging :: Maybe Text
    , parseAnsi :: Bool
    , plusEmphStyle :: Maybe Text
    , plusEmptyLineMarkerStyle :: Maybe Text
    , plusNonEmphStyle :: Maybe Text
    , plusStyle :: Maybe Text
    , raw :: Bool
    , relativePaths :: Bool
    , rightArrow :: Maybe Text
    , showColors :: Bool
    , showConfig :: Bool
    , showSyntaxThemes :: Bool
    , showThemes :: Bool
    , sideBySide :: Bool
    , syntaxTheme :: Maybe Text
    , tabs :: Maybe Int
    , trueColor :: Maybe Text
    , whitespaceErrorStyle :: Maybe Text
    , width :: Maybe Int
    , wordDiffRegex :: Maybe Text
    , wrapLeftSymbol :: Maybe Text
    , wrapMaxLines :: Maybe Int
    , wrapRightPercent :: Maybe Text
    , wrapRightPrefixSymbol :: Maybe Text
    , wrapRightSymbol :: Maybe Text
    , zeroStyle :: Maybe Text
    , opt24BitColor :: Maybe Text
    }
    deriving (Show, Eq)

-- | Default options - minimal flags, let the tool use its defaults
defaults :: Options
defaults =
    Options
        { blameCodeStyle = Nothing
        , blameFormat = Nothing
        , blamePalette = Nothing
        , blameSeparatorFormat = Nothing
        , blameSeparatorStyle = Nothing
        , blameTimestampFormat = Nothing
        , blameTimestampOutputFormat = Nothing
        , colorOnly = False
        , config = Nothing
        , commitDecorationStyle = Nothing
        , commitRegex = Nothing
        , commitStyle = Nothing
        , dark = False
        , defaultLanguage = Nothing
        , detectDarkLight = Nothing
        , diffHighlight = False
        , diffSoFancy = False
        , diffStatAlignWidth = Nothing
        , features = Nothing
        , fileAddedLabel = Nothing
        , fileCopiedLabel = Nothing
        , fileDecorationStyle = Nothing
        , fileModifiedLabel = Nothing
        , fileRemovedLabel = Nothing
        , fileRenamedLabel = Nothing
        , fileStyle = Nothing
        , fileTransformation = Nothing
        , generateCompletion = Nothing
        , grepContextLineStyle = Nothing
        , grepFileStyle = Nothing
        , grepHeaderDecorationStyle = Nothing
        , grepHeaderFileStyle = Nothing
        , grepLineNumberStyle = Nothing
        , grepOutputType = Nothing
        , grepMatchLineStyle = Nothing
        , grepMatchWordStyle = Nothing
        , grepSeparatorSymbol = Nothing
        , hunkHeaderDecorationStyle = Nothing
        , hunkHeaderFileStyle = Nothing
        , hunkHeaderLineNumberStyle = Nothing
        , hunkHeaderStyle = Nothing
        , hunkLabel = Nothing
        , hyperlinks = False
        , hyperlinksCommitLinkFormat = Nothing
        , hyperlinksFileLinkFormat = Nothing
        , inlineHintStyle = Nothing
        , inspectRawLines = Nothing
        , keepPlusMinusMarkers = False
        , light = False
        , lineBufferSize = Nothing
        , lineFillMethod = Nothing
        , lineNumbers = False
        , lineNumbersLeftFormat = Nothing
        , lineNumbersLeftStyle = Nothing
        , lineNumbersMinusStyle = Nothing
        , lineNumbersPlusStyle = Nothing
        , lineNumbersRightFormat = Nothing
        , lineNumbersRightStyle = Nothing
        , lineNumbersZeroStyle = Nothing
        , listLanguages = False
        , listSyntaxThemes = False
        , mapStyles = Nothing
        , maxLineDistance = Nothing
        , maxSyntaxHighlightingLength = Nothing
        , maxLineLength = Nothing
        , mergeConflictBeginSymbol = Nothing
        , mergeConflictEndSymbol = Nothing
        , mergeConflictOursDiffHeaderDecorationStyle = Nothing
        , mergeConflictOursDiffHeaderStyle = Nothing
        , mergeConflictTheirsDiffHeaderDecorationStyle = Nothing
        , mergeConflictTheirsDiffHeaderStyle = Nothing
        , minusEmptyLineMarkerStyle = Nothing
        , minusEmphStyle = Nothing
        , minusNonEmphStyle = Nothing
        , minusStyle = Nothing
        , navigate = False
        , navigateRegex = Nothing
        , noGitconfig = False
        , pager = Nothing
        , paging = Nothing
        , parseAnsi = False
        , plusEmphStyle = Nothing
        , plusEmptyLineMarkerStyle = Nothing
        , plusNonEmphStyle = Nothing
        , plusStyle = Nothing
        , raw = False
        , relativePaths = False
        , rightArrow = Nothing
        , showColors = False
        , showConfig = False
        , showSyntaxThemes = False
        , showThemes = False
        , sideBySide = False
        , syntaxTheme = Nothing
        , tabs = Nothing
        , trueColor = Nothing
        , whitespaceErrorStyle = Nothing
        , width = Nothing
        , wordDiffRegex = Nothing
        , wrapLeftSymbol = Nothing
        , wrapMaxLines = Nothing
        , wrapRightPercent = Nothing
        , wrapRightPrefixSymbol = Nothing
        , wrapRightSymbol = Nothing
        , zeroStyle = Nothing
        , opt24BitColor = Nothing
        }

-- | Build command-line arguments from options
buildArgs :: Options -> [Text]
buildArgs Options{..} =
    catMaybes
        [ opt blameCodeStyle "--blame-code-style"
        , opt blameFormat "--blame-format"
        , opt blamePalette "--blame-palette"
        , opt blameSeparatorFormat "--blame-separator-format"
        , opt blameSeparatorStyle "--blame-separator-style"
        , opt blameTimestampFormat "--blame-timestamp-format"
        , opt blameTimestampOutputFormat "--blame-timestamp-output-format"
        , flag colorOnly "--color-only"
        , optShow config "--config"
        , opt commitDecorationStyle "--commit-decoration-style"
        , opt commitRegex "--commit-regex"
        , opt commitStyle "--commit-style"
        , flag dark "--dark"
        , opt defaultLanguage "--default-language"
        , opt detectDarkLight "--detect-dark-light"
        , flag diffHighlight "--diff-highlight"
        , flag diffSoFancy "--diff-so-fancy"
        , optShow diffStatAlignWidth "--diff-stat-align-width"
        , opt features "--features"
        , opt fileAddedLabel "--file-added-label"
        , opt fileCopiedLabel "--file-copied-label"
        , opt fileDecorationStyle "--file-decoration-style"
        , opt fileModifiedLabel "--file-modified-label"
        , opt fileRemovedLabel "--file-removed-label"
        , opt fileRenamedLabel "--file-renamed-label"
        , opt fileStyle "--file-style"
        , opt fileTransformation "--file-transformation"
        , opt generateCompletion "--generate-completion"
        , opt grepContextLineStyle "--grep-context-line-style"
        , opt grepFileStyle "--grep-file-style"
        , opt grepHeaderDecorationStyle "--grep-header-decoration-style"
        , opt grepHeaderFileStyle "--grep-header-file-style"
        , opt grepLineNumberStyle "--grep-line-number-style"
        , opt grepOutputType "--grep-output-type"
        , opt grepMatchLineStyle "--grep-match-line-style"
        , opt grepMatchWordStyle "--grep-match-word-style"
        , opt grepSeparatorSymbol "--grep-separator-symbol"
        , opt hunkHeaderDecorationStyle "--hunk-header-decoration-style"
        , opt hunkHeaderFileStyle "--hunk-header-file-style"
        , opt hunkHeaderLineNumberStyle "--hunk-header-line-number-style"
        , opt hunkHeaderStyle "--hunk-header-style"
        , opt hunkLabel "--hunk-label"
        , flag hyperlinks "--hyperlinks"
        , opt hyperlinksCommitLinkFormat "--hyperlinks-commit-link-format"
        , opt hyperlinksFileLinkFormat "--hyperlinks-file-link-format"
        , opt inlineHintStyle "--inline-hint-style"
        , opt inspectRawLines "--inspect-raw-lines"
        , flag keepPlusMinusMarkers "--keep-plus-minus-markers"
        , flag light "--light"
        , optShow lineBufferSize "--line-buffer-size"
        , opt lineFillMethod "--line-fill-method"
        , flag lineNumbers "--line-numbers"
        , opt lineNumbersLeftFormat "--line-numbers-left-format"
        , opt lineNumbersLeftStyle "--line-numbers-left-style"
        , opt lineNumbersMinusStyle "--line-numbers-minus-style"
        , opt lineNumbersPlusStyle "--line-numbers-plus-style"
        , opt lineNumbersRightFormat "--line-numbers-right-format"
        , opt lineNumbersRightStyle "--line-numbers-right-style"
        , opt lineNumbersZeroStyle "--line-numbers-zero-style"
        , flag listLanguages "--list-languages"
        , flag listSyntaxThemes "--list-syntax-themes"
        , opt mapStyles "--map-styles"
        , opt maxLineDistance "--max-line-distance"
        , optShow maxSyntaxHighlightingLength "--max-syntax-highlighting-length"
        , optShow maxLineLength "--max-line-length"
        , opt mergeConflictBeginSymbol "--merge-conflict-begin-symbol"
        , opt mergeConflictEndSymbol "--merge-conflict-end-symbol"
        , opt mergeConflictOursDiffHeaderDecorationStyle "--merge-conflict-ours-diff-header-decoration-style"
        , opt mergeConflictOursDiffHeaderStyle "--merge-conflict-ours-diff-header-style"
        , opt mergeConflictTheirsDiffHeaderDecorationStyle "--merge-conflict-theirs-diff-header-decoration-style"
        , opt mergeConflictTheirsDiffHeaderStyle "--merge-conflict-theirs-diff-header-style"
        , opt minusEmptyLineMarkerStyle "--minus-empty-line-marker-style"
        , opt minusEmphStyle "--minus-emph-style"
        , opt minusNonEmphStyle "--minus-non-emph-style"
        , opt minusStyle "--minus-style"
        , flag navigate "--navigate"
        , opt navigateRegex "--navigate-regex"
        , flag noGitconfig "--no-gitconfig"
        , opt pager "--pager"
        , opt paging "--paging"
        , flag parseAnsi "--parse-ansi"
        , opt plusEmphStyle "--plus-emph-style"
        , opt plusEmptyLineMarkerStyle "--plus-empty-line-marker-style"
        , opt plusNonEmphStyle "--plus-non-emph-style"
        , opt plusStyle "--plus-style"
        , flag raw "--raw"
        , flag relativePaths "--relative-paths"
        , opt rightArrow "--right-arrow"
        , flag showColors "--show-colors"
        , flag showConfig "--show-config"
        , flag showSyntaxThemes "--show-syntax-themes"
        , flag showThemes "--show-themes"
        , flag sideBySide "--side-by-side"
        , opt syntaxTheme "--syntax-theme"
        , optShow tabs "--tabs"
        , opt trueColor "--true-color"
        , opt whitespaceErrorStyle "--whitespace-error-style"
        , optShow width "--width"
        , opt wordDiffRegex "--word-diff-regex"
        , opt wrapLeftSymbol "--wrap-left-symbol"
        , optShow wrapMaxLines "--wrap-max-lines"
        , opt wrapRightPercent "--wrap-right-percent"
        , opt wrapRightPrefixSymbol "--wrap-right-prefix-symbol"
        , opt wrapRightSymbol "--wrap-right-symbol"
        , opt zeroStyle "--zero-style"
        , opt opt24BitColor "--24-bit-color"
        ]
  where
    flag True f = Just f
    flag False _ = Nothing
    opt (Just v) f = Just (f <> "=" <> v)
    opt Nothing _ = Nothing
    optShow (Just v) f = Just (f <> "=" <> pack (show v))
    optShow Nothing _ = Nothing

{- | Run delta with options and additional arguments

Returns stdout. Throws on non-zero exit.
-}
delta :: Options -> [Text] -> Sh Text
delta opts args = run "delta" (buildArgs opts ++ args)

-- | Run delta, ignoring output
delta_ :: Options -> [Text] -> Sh ()
delta_ opts args = run_ "delta" (buildArgs opts ++ args)
