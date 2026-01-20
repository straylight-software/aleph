{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- | Generated typed wrapper for rg

This module was auto-generated from --help output.
Do not edit manually.
-}
module Rg (
    RgOptions (..),
    defaultRgOptions,
    runRg,
    runRg_,
) where

import Aleph.Script
import Data.Text (Text)
import qualified Data.Text as T

-- | Options for Rg
data RgOptions = RgOptions
    { optRegexp :: Maybe Text
    -- ^ A pattern to search for.
    , optFile :: Maybe Text
    -- ^ Search for patterns from the given file.
    , optPre :: Maybe Text
    -- ^ Search output of COMMAND for each PATH.
    , optPreGlob :: Maybe Text
    -- ^ Include or exclude files from a preprocessor.
    , optSearchZip :: Bool
    -- ^ Search in compressed files.
    , optCaseSensitive :: Bool
    -- ^ Search case sensitively (default).
    , optCrlf :: Bool
    -- ^ Use CRLF line terminators (nice for Windows).
    , optDfaSizeLimit :: Maybe Int
    -- ^ The upper size limit of the regex DFA.
    , optEncoding :: Maybe Text
    -- ^ Specify the text encoding of files to search.
    , optEngine :: Maybe Text
    -- ^ Specify which regex engine to use.
    , optFixedStrings :: Bool
    -- ^ Treat all patterns as literals.
    , optIgnoreCase :: Bool
    -- ^ Case insensitive search.
    , optInvertMatch :: Bool
    -- ^ Invert matching.
    , optLineRegexp :: Bool
    -- ^ Show matches surrounded by line boundaries.
    , optMaxCount :: Maybe Int
    -- ^ Limit the number of matching lines.
    , optMmap :: Bool
    -- ^ Search with memory maps when possible.
    , optMultiline :: Bool
    -- ^ Enable searching across multiple lines.
    , optMultilineDotall :: Bool
    -- ^ Make '.' match line terminators.
    , optNoUnicode :: Bool
    -- ^ Disable Unicode mode.
    , optNullData :: Bool
    -- ^ Use NUL as a line terminator.
    , optPcre2 :: Bool
    -- ^ Enable PCRE2 matching.
    , optRegexSizeLimit :: Maybe Int
    -- ^ The size limit of the compiled regex.
    , optSmartCase :: Bool
    -- ^ Smart case search.
    , optStopOnNonmatch :: Bool
    -- ^ Stop searching after a non-match.
    , optText :: Bool
    -- ^ Search binary files as if they were text.
    , optThreads :: Maybe Int
    -- ^ Set the approximate number of threads to use.
    , optWordRegexp :: Bool
    -- ^ Show matches surrounded by word boundaries.
    , optAutoHybridRegex :: Bool
    -- ^ (DEPRECATED) Use PCRE2 if appropriate.
    , optNoPcre2Unicode :: Bool
    -- ^ (DEPRECATED) Disable Unicode mode for PCRE2.
    , optBinary :: Bool
    -- ^ Search binary files.
    , optFollow :: Bool
    -- ^ Follow symbolic links.
    , optGlob :: Maybe Text
    -- ^ Include or exclude file paths.
    , optGlobCaseInsensitive :: Bool
    -- ^ Process all glob patterns case insensitively.
    , optHidden :: Bool
    -- ^ Search hidden files and directories.
    , optIglob :: Maybe Text
    -- ^ Include/exclude paths case insensitively.
    , optIgnoreFile :: Maybe FilePath
    -- ^ Specify additional ignore files.
    , optIgnoreFileCaseInsensitive :: Bool
    -- ^ Process ignore files case insensitively.
    , optMaxDepth :: Maybe Int
    -- ^ Descend at most NUM directories.
    , optMaxFilesize :: Maybe Int
    -- ^ Ignore files larger than NUM in size.
    , optNoIgnore :: Bool
    -- ^ Don't use ignore files.
    , optNoIgnoreDot :: Bool
    -- ^ Don't use .ignore or .rgignore files.
    , optNoIgnoreExclude :: Bool
    -- ^ Don't use local exclusion files.
    , optNoIgnoreFiles :: Bool
    -- ^ Don't use --ignore-file arguments.
    , optNoIgnoreGlobal :: Bool
    -- ^ Don't use global ignore files.
    , optNoIgnoreParent :: Bool
    -- ^ Don't use ignore files in parent directories.
    , optNoIgnoreVcs :: Bool
    -- ^ Don't use ignore files from source control.
    , optNoRequireGit :: Bool
    -- ^ Use .gitignore outside of git repositories.
    , optOneFileSystem :: Bool
    -- ^ Skip directories on other file systems.
    , optType :: Maybe Text
    -- ^ Only search files matching TYPE.
    , optTypeNot :: Maybe Text
    -- ^ Do not search files matching TYPE.
    , optTypeAdd :: Maybe Text
    -- ^ Add a new glob for a file type.
    , optTypeClear :: Maybe Text
    -- ^ Clear globs for a file type.
    , optUnrestricted :: Bool
    -- ^ Reduce the level of "smart" filtering.
    , optAfterContext :: Maybe Int
    -- ^ Show NUM lines after each match.
    , optBeforeContext :: Maybe Int
    -- ^ Show NUM lines before each match.
    , optBlockBuffered :: Bool
    -- ^ Force block buffering.
    , optByteOffset :: Bool
    -- ^ Print the byte offset for each matching line.
    , optColor :: Maybe Text
    -- ^ When to use color.
    , optColors :: Maybe Text
    -- ^ Configure color settings and styles.
    , optColumn :: Bool
    -- ^ Show column numbers.
    , optContext :: Maybe Int
    -- ^ Show NUM lines before and after each match.
    , optContextSeparator :: Maybe Text
    -- ^ Set the separator for contextual chunks.
    , optFieldContextSeparator :: Maybe Text
    -- ^ Set the field context separator.
    , optFieldMatchSeparator :: Maybe Text
    -- ^ Set the field match separator.
    , optHeading :: Bool
    -- ^ Print matches grouped by each file.
    , optHostnameBin :: Maybe Text
    -- ^ Run a program to get this system's hostname.
    , optHyperlinkFormat :: Maybe Text
    -- ^ Set the format of hyperlinks.
    , optIncludeZero :: Bool
    -- ^ Include zero matches in summary output.
    , optLineBuffered :: Bool
    -- ^ Force line buffering.
    , optLineNumber :: Bool
    -- ^ Show line numbers.
    , optNoLineNumber :: Bool
    -- ^ Suppress line numbers.
    , optMaxColumns :: Maybe Int
    -- ^ Omit lines longer than this limit.
    , optMaxColumnsPreview :: Bool
    -- ^ Show preview for lines exceeding the limit.
    , optNull :: Bool
    -- ^ Print a NUL byte after file paths.
    , optOnlyMatching :: Bool
    -- ^ Print only matched parts of a line.
    , optPathSeparator :: Maybe Text
    -- ^ Set the path separator for printing paths.
    , optPassthru :: Bool
    -- ^ Print both matching and non-matching lines.
    , optPretty :: Bool
    -- ^ Alias for colors, headings and line numbers.
    , optQuiet :: Bool
    -- ^ Do not print anything to stdout.
    , optReplace :: Maybe Text
    -- ^ Replace matches with the given text.
    , optSort :: Maybe Text
    -- ^ Sort results in ascending order.
    , optSortr :: Maybe Text
    -- ^ Sort results in descending order.
    , optTrim :: Bool
    -- ^ Trim prefix whitespace from matches.
    , optVimgrep :: Bool
    -- ^ Print results in a vim compatible format.
    , optWithFilename :: Bool
    -- ^ Print the file path with each matching line.
    , optNoFilename :: Bool
    -- ^ Never print the path with each matching line.
    , optSortFiles :: Bool
    -- ^ (DEPRECATED) Sort results by file path.
    , optCount :: Bool
    -- ^ Show count of matching lines for each file.
    , optCountMatches :: Bool
    -- ^ Show count of every match for each file.
    , optFilesWithMatches :: Bool
    -- ^ Print the paths with at least one match.
    , optFilesWithoutMatch :: Bool
    -- ^ Print the paths that contain zero matches.
    , optJson :: Bool
    -- ^ Show search results in a JSON Lines format.
    , optDebug :: Bool
    -- ^ Show debug messages.
    , optNoIgnoreMessages :: Bool
    -- ^ Suppress gitignore parse error messages.
    , optNoMessages :: Bool
    -- ^ Suppress some error messages.
    , optStats :: Bool
    -- ^ Print statistics about the search.
    , optTrace :: Bool
    -- ^ Show trace messages.
    , optFiles :: Bool
    -- ^ Print each file that would be searched.
    , optGenerate :: Maybe Text
    -- ^ Generate man pages and completion scripts.
    , optNoConfig :: Bool
    -- ^ Never read configuration files.
    , optPcre2Version :: Bool
    -- ^ Print the version of PCRE2 that ripgrep uses.
    , optTypeList :: Bool
    -- ^ Show all supported file types.
    }
    deriving (Show, Eq)

-- | Default options (all Nothing/False)
defaultRgOptions :: RgOptions
defaultRgOptions =
    RgOptions
        { optRegexp = Nothing
        , optFile = Nothing
        , optPre = Nothing
        , optPreGlob = Nothing
        , optSearchZip = False
        , optCaseSensitive = False
        , optCrlf = False
        , optDfaSizeLimit = Nothing
        , optEncoding = Nothing
        , optEngine = Nothing
        , optFixedStrings = False
        , optIgnoreCase = False
        , optInvertMatch = False
        , optLineRegexp = False
        , optMaxCount = Nothing
        , optMmap = False
        , optMultiline = False
        , optMultilineDotall = False
        , optNoUnicode = False
        , optNullData = False
        , optPcre2 = False
        , optRegexSizeLimit = Nothing
        , optSmartCase = False
        , optStopOnNonmatch = False
        , optText = False
        , optThreads = Nothing
        , optWordRegexp = False
        , optAutoHybridRegex = False
        , optNoPcre2Unicode = False
        , optBinary = False
        , optFollow = False
        , optGlob = Nothing
        , optGlobCaseInsensitive = False
        , optHidden = False
        , optIglob = Nothing
        , optIgnoreFile = Nothing
        , optIgnoreFileCaseInsensitive = False
        , optMaxDepth = Nothing
        , optMaxFilesize = Nothing
        , optNoIgnore = False
        , optNoIgnoreDot = False
        , optNoIgnoreExclude = False
        , optNoIgnoreFiles = False
        , optNoIgnoreGlobal = False
        , optNoIgnoreParent = False
        , optNoIgnoreVcs = False
        , optNoRequireGit = False
        , optOneFileSystem = False
        , optType = Nothing
        , optTypeNot = Nothing
        , optTypeAdd = Nothing
        , optTypeClear = Nothing
        , optUnrestricted = False
        , optAfterContext = Nothing
        , optBeforeContext = Nothing
        , optBlockBuffered = False
        , optByteOffset = False
        , optColor = Nothing
        , optColors = Nothing
        , optColumn = False
        , optContext = Nothing
        , optContextSeparator = Nothing
        , optFieldContextSeparator = Nothing
        , optFieldMatchSeparator = Nothing
        , optHeading = False
        , optHostnameBin = Nothing
        , optHyperlinkFormat = Nothing
        , optIncludeZero = False
        , optLineBuffered = False
        , optLineNumber = False
        , optNoLineNumber = False
        , optMaxColumns = Nothing
        , optMaxColumnsPreview = False
        , optNull = False
        , optOnlyMatching = False
        , optPathSeparator = Nothing
        , optPassthru = False
        , optPretty = False
        , optQuiet = False
        , optReplace = Nothing
        , optSort = Nothing
        , optSortr = Nothing
        , optTrim = False
        , optVimgrep = False
        , optWithFilename = False
        , optNoFilename = False
        , optSortFiles = False
        , optCount = False
        , optCountMatches = False
        , optFilesWithMatches = False
        , optFilesWithoutMatch = False
        , optJson = False
        , optDebug = False
        , optNoIgnoreMessages = False
        , optNoMessages = False
        , optStats = False
        , optTrace = False
        , optFiles = False
        , optGenerate = Nothing
        , optNoConfig = False
        , optPcre2Version = False
        , optTypeList = False
        }

-- | Run rg with the given options, capturing stdout
runRg :: RgOptions -> [Text] -> Sh Text
runRg opts args = run "rg" (buildArgs opts ++ args)

-- | Run rg with the given options, ignoring output
runRg_ :: RgOptions -> [Text] -> Sh ()
runRg_ opts args = run_ "rg" (buildArgs opts ++ args)

-- | Build command-line arguments from options
buildArgs :: RgOptions -> [Text]
buildArgs RgOptions{..} =
    concat
        [ maybe [] (\v -> ["--regexp", v]) optRegexp
        , maybe [] (\v -> ["--file", v]) optFile
        , maybe [] (\v -> ["--pre", v]) optPre
        , maybe [] (\v -> ["--pre-glob", v]) optPreGlob
        , if optSearchZip then ["--search-zip"] else []
        , if optCaseSensitive then ["--case-sensitive"] else []
        , if optCrlf then ["--crlf"] else []
        , maybe [] (\v -> ["--dfa-size-limit", T.pack (show v)]) optDfaSizeLimit
        , maybe [] (\v -> ["--encoding", v]) optEncoding
        , maybe [] (\v -> ["--engine", v]) optEngine
        , if optFixedStrings then ["--fixed-strings"] else []
        , if optIgnoreCase then ["--ignore-case"] else []
        , if optInvertMatch then ["--invert-match"] else []
        , if optLineRegexp then ["--line-regexp"] else []
        , maybe [] (\v -> ["--max-count", T.pack (show v)]) optMaxCount
        , if optMmap then ["--mmap"] else []
        , if optMultiline then ["--multiline"] else []
        , if optMultilineDotall then ["--multiline-dotall"] else []
        , if optNoUnicode then ["--no-unicode"] else []
        , if optNullData then ["--null-data"] else []
        , if optPcre2 then ["--pcre2"] else []
        , maybe [] (\v -> ["--regex-size-limit", T.pack (show v)]) optRegexSizeLimit
        , if optSmartCase then ["--smart-case"] else []
        , if optStopOnNonmatch then ["--stop-on-nonmatch"] else []
        , if optText then ["--text"] else []
        , maybe [] (\v -> ["--threads", T.pack (show v)]) optThreads
        , if optWordRegexp then ["--word-regexp"] else []
        , if optAutoHybridRegex then ["--auto-hybrid-regex"] else []
        , if optNoPcre2Unicode then ["--no-pcre2-unicode"] else []
        , if optBinary then ["--binary"] else []
        , if optFollow then ["--follow"] else []
        , maybe [] (\v -> ["--glob", v]) optGlob
        , if optGlobCaseInsensitive then ["--glob-case-insensitive"] else []
        , if optHidden then ["--hidden"] else []
        , maybe [] (\v -> ["--iglob", v]) optIglob
        , maybe [] (\v -> ["--ignore-file", T.pack (show v)]) optIgnoreFile
        , if optIgnoreFileCaseInsensitive then ["--ignore-file-case-insensitive"] else []
        , maybe [] (\v -> ["--max-depth", T.pack (show v)]) optMaxDepth
        , maybe [] (\v -> ["--max-filesize", T.pack (show v)]) optMaxFilesize
        , if optNoIgnore then ["--no-ignore"] else []
        , if optNoIgnoreDot then ["--no-ignore-dot"] else []
        , if optNoIgnoreExclude then ["--no-ignore-exclude"] else []
        , if optNoIgnoreFiles then ["--no-ignore-files"] else []
        , if optNoIgnoreGlobal then ["--no-ignore-global"] else []
        , if optNoIgnoreParent then ["--no-ignore-parent"] else []
        , if optNoIgnoreVcs then ["--no-ignore-vcs"] else []
        , if optNoRequireGit then ["--no-require-git"] else []
        , if optOneFileSystem then ["--one-file-system"] else []
        , maybe [] (\v -> ["--type", v]) optType
        , maybe [] (\v -> ["--type-not", v]) optTypeNot
        , maybe [] (\v -> ["--type-add", v]) optTypeAdd
        , maybe [] (\v -> ["--type-clear", v]) optTypeClear
        , if optUnrestricted then ["--unrestricted"] else []
        , maybe [] (\v -> ["--after-context", T.pack (show v)]) optAfterContext
        , maybe [] (\v -> ["--before-context", T.pack (show v)]) optBeforeContext
        , if optBlockBuffered then ["--block-buffered"] else []
        , if optByteOffset then ["--byte-offset"] else []
        , maybe [] (\v -> ["--color", v]) optColor
        , maybe [] (\v -> ["--colors", v]) optColors
        , if optColumn then ["--column"] else []
        , maybe [] (\v -> ["--context", T.pack (show v)]) optContext
        , maybe [] (\v -> ["--context-separator", v]) optContextSeparator
        , maybe [] (\v -> ["--field-context-separator", v]) optFieldContextSeparator
        , maybe [] (\v -> ["--field-match-separator", v]) optFieldMatchSeparator
        , if optHeading then ["--heading"] else []
        , maybe [] (\v -> ["--hostname-bin", v]) optHostnameBin
        , maybe [] (\v -> ["--hyperlink-format", v]) optHyperlinkFormat
        , if optIncludeZero then ["--include-zero"] else []
        , if optLineBuffered then ["--line-buffered"] else []
        , if optLineNumber then ["--line-number"] else []
        , if optNoLineNumber then ["--no-line-number"] else []
        , maybe [] (\v -> ["--max-columns", T.pack (show v)]) optMaxColumns
        , if optMaxColumnsPreview then ["--max-columns-preview"] else []
        , if optNull then ["--null"] else []
        , if optOnlyMatching then ["--only-matching"] else []
        , maybe [] (\v -> ["--path-separator", v]) optPathSeparator
        , if optPassthru then ["--passthru"] else []
        , if optPretty then ["--pretty"] else []
        , if optQuiet then ["--quiet"] else []
        , maybe [] (\v -> ["--replace", v]) optReplace
        , maybe [] (\v -> ["--sort", v]) optSort
        , maybe [] (\v -> ["--sortr", v]) optSortr
        , if optTrim then ["--trim"] else []
        , if optVimgrep then ["--vimgrep"] else []
        , if optWithFilename then ["--with-filename"] else []
        , if optNoFilename then ["--no-filename"] else []
        , if optSortFiles then ["--sort-files"] else []
        , if optCount then ["--count"] else []
        , if optCountMatches then ["--count-matches"] else []
        , if optFilesWithMatches then ["--files-with-matches"] else []
        , if optFilesWithoutMatch then ["--files-without-match"] else []
        , if optJson then ["--json"] else []
        , if optDebug then ["--debug"] else []
        , if optNoIgnoreMessages then ["--no-ignore-messages"] else []
        , if optNoMessages then ["--no-messages"] else []
        , if optStats then ["--stats"] else []
        , if optTrace then ["--trace"] else []
        , if optFiles then ["--files"] else []
        , maybe [] (\v -> ["--generate", v]) optGenerate
        , if optNoConfig then ["--no-config"] else []
        , if optPcre2Version then ["--pcre2-version"] else []
        , if optTypeList then ["--type-list"] else []
        ]
