{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Clap
Description : Parser and code generator for clap CLI tools

Parse --help output from clap-based CLI tools and generate typed
Haskell wrappers for invoking them safely.

Example usage:

> -- Parse ripgrep help and generate code
> import Aleph.Script.Clap
>
> main = do
>   help <- readProcess "rg" ["--help"] ""
>   let parsed = parseHelp (T.pack help)
>   TIO.putStrLn $ generateModule "Rg" "rg" parsed
-}
module Aleph.Script.Clap (
    -- * Types
    ClapOption (..),
    ClapPositional (..),
    ClapSection (..),
    ClapHelp (..),

    -- * Parsing
    parseHelp,

    -- * Code generation
    generateModule,
    generateOptions,
    generateDefaults,
    generateBuildArgs,
    generateInvoke,

    -- * Utilities
    optionToHaskellName,
    valueNameToType,
) where

import Control.Monad (void)
import Data.Char (isAlphaNum, toUpper)
import Data.Text (Text)
import qualified Data.Text as T
import Data.Void
import Text.Megaparsec
import Text.Megaparsec.Char (char, digitChar, letterChar, string)

-- ============================================================================
-- Types
-- ============================================================================

-- | A CLI option/flag
data ClapOption = ClapOption
    { optShort :: Maybe Char
    -- ^ Short form: -x
    , optLong :: Maybe Text
    -- ^ Long form: --long-name
    , optArg :: Maybe Text
    -- ^ Argument placeholder: FILE, NUM, etc.
    , optDesc :: Text
    -- ^ Description
    }
    deriving (Show, Eq)

-- | A positional argument
data ClapPositional = ClapPositional
    { posName :: Text
    -- ^ Name: FILE, PATTERN, etc.
    , posRequired :: Bool
    -- ^ Required (<FILE>) vs optional ([FILE])
    }
    deriving (Show, Eq)

-- | A section of the help output
data ClapSection = ClapSection
    { secName :: Text
    -- ^ Section name: "Options", "Input", etc.
    , secOptions :: [ClapOption]
    -- ^ Options in this section
    }
    deriving (Show, Eq)

-- | Full parsed help
data ClapHelp = ClapHelp
    { helpSections :: [ClapSection]
    -- ^ All sections
    , helpPositionals :: [ClapPositional]
    -- ^ Positional arguments
    }
    deriving (Show, Eq)

-- ============================================================================
-- Parser
-- ============================================================================

type Parser = Parsec Void Text

hspace :: Parser ()
hspace = void $ takeWhileP Nothing (\c -> c == ' ' || c == '\t')

restOfLine :: Parser Text
restOfLine = do
    content <- takeWhileP Nothing (/= '\n')
    void (char '\n') <|> eof
    return content

shortOpt :: Parser Char
shortOpt = char '-' *> (letterChar <|> digitChar <|> char '.')

longOpt :: Parser Text
longOpt = string "--" *> takeWhile1P Nothing isOptChar
  where
    isOptChar c = c == '-' || c == '_' || c `elem` ['a' .. 'z'] || c `elem` ['A' .. 'Z'] || c `elem` ['0' .. '9']

argPlaceholder :: Parser Text
argPlaceholder =
    choice
        [ char '=' *> takeWhile1P Nothing (\c -> c /= ' ' && c /= '\t' && c /= '\n')
        , try $ hspace *> char '<' *> takeWhile1P Nothing (/= '>') <* char '>'
        , char '[' *> (char '=' *> takeWhile1P Nothing (\c -> c /= ']' && c /= '\n')) <* char ']'
        ]

{- | Parse an option line
Clap option lines have specific indentation:
  - 2-6 spaces before the first dash
  - Description continuation lines have 8+ spaces

Handles both short and long help formats:
  - Short: "-e, --regexp=PATTERN"
  - Long:  "-e PATTERN, --regexp=PATTERN"
-}
optionLine :: Parser ClapOption
optionLine = do
    -- Count leading spaces - option lines have 2-6 spaces
    spaces <- takeWhileP Nothing (== ' ')
    let indent = T.length spaces
    -- Option lines have modest indentation (2-6 spaces typically)
    -- Description continuation lines have more (8+ spaces)
    if indent < 2 || indent > 6
        then empty
        else do
            -- Parse short option with optional value placeholder
            mShort <- optional $ try shortOpt
            -- In long help format, value placeholder can come after short option
            mShortArg <- case mShort of
                Just _ -> optional $ try $ do
                    hspace
                    -- Uppercase word is a value placeholder (e.g., "PATTERN")
                    takeWhile1P Nothing (\c -> c `elem` ['A' .. 'Z'] || c == '-' || c == '_')
                Nothing -> return Nothing
            -- Skip comma separator between short and long
            _ <- optional $ try $ do
                _ <- optional hspace
                _ <- char ','
                hspace
            -- Parse long option
            mLong <- optional $ try longOpt
            -- Parse value placeholder after long option (=VALUE or <VALUE>)
            mLongArg <- optional $ try argPlaceholder
            hspace
            desc <- restOfLine
            case (mShort, mLong) of
                (Nothing, Nothing) -> empty
                _ ->
                    return
                        ClapOption
                            { optShort = mShort
                            , optLong = mLong
                            , -- Prefer long option's arg, fall back to short option's arg
                              optArg = mLongArg <|> mShortArg
                            , optDesc = T.strip desc
                            }

positionalLine :: Parser ClapPositional
positionalLine = do
    hspace
    (name, req) <-
        choice
            [ do
                _ <- char '<'
                n <- takeWhile1P Nothing (\c -> c /= '>' && c /= '\n')
                _ <- char '>'
                return (n, True)
            , do
                _ <- char '['
                n <- takeWhile1P Nothing (\c -> c /= ']' && c /= '\n')
                _ <- char ']'
                return (n, False)
            ]
    _ <- restOfLine
    return ClapPositional{posName = name, posRequired = req}

isSectionLine :: Text -> Bool
isSectionLine t =
    let stripped = T.strip t
     in not (T.null stripped)
            && T.last stripped == ':'
            && not (T.isPrefixOf "-" stripped)
            && not (T.isPrefixOf " " t)
            && not (T.isInfixOf "://" stripped)

-- | Parse clap help output into structured form
parseHelp :: Text -> ClapHelp
parseHelp input =
    let lns = T.lines input
        (sections, positionals) = go [] [] Nothing lns
     in ClapHelp{helpSections = sections, helpPositionals = positionals}
  where
    go secAcc posAcc _curSec [] =
        case _curSec of
            Just (name, opts) -> (reverse (ClapSection name (reverse opts) : secAcc), reverse posAcc)
            Nothing -> (reverse secAcc, reverse posAcc)
    go secAcc posAcc curSec (line : rest)
        | isSectionLine line =
            let secName = T.dropEnd 1 (T.strip line)
                secAcc' = case curSec of
                    Just (name, opts) -> ClapSection name (reverse opts) : secAcc
                    Nothing -> secAcc
             in go secAcc' posAcc (Just (secName, [])) rest
        | Just (name, opts) <- curSec =
            if name == "Arguments" || name == "POSITIONAL ARGUMENTS"
                then case parse positionalLine "" (line <> "\n") of
                    Right pos -> go secAcc (pos : posAcc) curSec rest
                    Left _ ->
                        if T.null (T.strip line) || T.isPrefixOf "  " line
                            then go secAcc posAcc curSec rest
                            else go (ClapSection name (reverse opts) : secAcc) posAcc Nothing rest
                else case parse optionLine "" (line <> "\n") of
                    Right opt -> go secAcc posAcc (Just (name, opt : opts)) rest
                    Left _ ->
                        if T.null (T.strip line) || T.isPrefixOf "  " line
                            then go secAcc posAcc curSec rest
                            else go (ClapSection name (reverse opts) : secAcc) posAcc Nothing rest
        | otherwise = go secAcc posAcc curSec rest

-- ============================================================================
-- Code Generation Utilities
-- ============================================================================

{- | Convert option long name to valid Haskell identifier
e.g., "ignore-case" -> "ignoreCase"
      "24-bit-color" -> "opt24BitColor" (prefixed because starts with digit)
-}
optionToHaskellName :: Text -> Text
optionToHaskellName = ensureValidStart . toCamelCase . T.filter isValidChar
  where
    isValidChar c = isAlphaNum c || c == '-' || c == '_'

    toCamelCase :: Text -> Text
    toCamelCase t = case T.splitOn "-" t of
        [] -> "opt"
        (first : rest) -> T.concat $ T.toLower first : map capitalize rest

    capitalize :: Text -> Text
    capitalize t = case T.uncons t of
        Nothing -> t
        Just (c, cs) -> T.cons (toUpper c) cs

    -- Haskell identifiers can't start with a digit
    ensureValidStart :: Text -> Text
    ensureValidStart t = case T.uncons t of
        Just (c, _) | c >= '0' && c <= '9' -> "opt" <> capitalize t
        _ -> t

{- | Convert value name placeholder to a Haskell type hint
This is approximate - we can't know the exact type from just a placeholder
-}
valueNameToType :: Text -> Text
valueNameToType name = case T.toUpper name of
    "FILE" -> "FilePath"
    "PATH" -> "FilePath"
    "DIR" -> "FilePath"
    "NUM" -> "Int"
    "COUNT" -> "Int"
    "SIZE" -> "Int"
    "N" -> "Int"
    "SECONDS" -> "Double"
    "PATTERN" -> "Text"
    "REGEX" -> "Text"
    "GLOB" -> "Text"
    "STRING" -> "Text"
    "NAME" -> "Text"
    "URL" -> "Text"
    "WHEN" -> "Text" -- Usually an enum: always/auto/never
    "FORMAT" -> "Text"
    "LEVEL" -> "Text"
    "MODE" -> "Text"
    "TYPE" -> "Text"
    _ -> "Text" -- Default to Text for unknown types

-- ============================================================================
-- Code Generation
-- ============================================================================

{- | Generate a complete Haskell module for a CLI tool

The generated module follows the style of hand-crafted wrappers:

  * Clean field names (e.g., @ignoreCase@, not @optIgnoreCase@)
  * Options grouped by section with Haddock comments
  * Simple @defaults@ value and @cmd@/@cmd_@ invocation functions
  * Uses @catMaybes@ pattern for building args
-}
generateModule :: Text -> Text -> ClapHelp -> Text
generateModule moduleName cmdName help =
    let allOpts = concatMap secOptions (helpSections help)
        hasFilePath = any usesFilePath allOpts
        usesFilePath opt = case optArg opt of
            Just arg -> valueNameToType arg == "FilePath"
            Nothing -> False
        importLine =
            if hasFilePath
                then "import Aleph.Script hiding (FilePath)"
                else "import Aleph.Script"
     in T.unlines
            [ "{-# LANGUAGE OverloadedStrings #-}"
            , "{-# LANGUAGE RecordWildCards #-}"
            , "-- |"
            , "-- Module      : Aleph.Script.Tools." <> moduleName
            , "-- Description : Typed wrapper for " <> cmdName
            , "--"
            , "-- This module was auto-generated from @" <> cmdName <> " --help@ output."
            , "-- Review and adjust field names and types as needed."
            , "--"
            , "module Aleph.Script.Tools." <> moduleName
            , "  ( -- * Options"
            , "    Options(..)"
            , "  , defaults"
            , ""
            , "    -- * Invocation"
            , "  , " <> T.toLower moduleName
            , "  , " <> T.toLower moduleName <> "_"
            , "  ) where"
            , ""
            , importLine
            , "import Data.Maybe (catMaybes)"
            , ""
            , generateOptions help
            , ""
            , generateDefaults help
            , ""
            , generateBuildArgs help
            , ""
            , generateInvoke moduleName cmdName help
            ]

-- | Generate the options record type with sections preserved as comments
generateOptions :: ClapHelp -> Text
generateOptions ClapHelp{..} =
    T.unlines $
        [ "-- | Options record"
        , "--"
        , "-- Use 'defaults' and override fields as needed:"
        , "--"
        , "-- > defaults { ignoreCase = True, hidden = True }"
        , "--"
        , "data Options = Options"
        ]
            ++ fieldDefs
            ++ [ "  } deriving (Show, Eq)"
               ]
  where
    -- Filter out help/version, dedupe by field name across all sections
    allOptsDeduped = dedupeByName $ filter (not . isBuiltinOpt) $ concatMap secOptions helpSections

    isBuiltinOpt opt = optLong opt `elem` [Just "help", Just "version"]

    -- Generate field definitions (sections flattened after dedup)
    fieldDefs = case allOptsDeduped of
        [] -> ["  { _placeholder :: ()  -- ^ No options found"]
        (first : rest) ->
            ("  { " <> optField first) : map (\o -> "  , " <> optField o) rest

    optField opt =
        let name = fieldName opt
            typ = fieldType opt
            short = maybe "" (\c -> "-" <> T.singleton c <> ": ") (optShort opt)
            desc = T.take 60 (optDesc opt)
            comment = if T.null desc then "" else " -- ^ " <> short <> desc
         in name <> " :: " <> typ <> comment

    fieldType opt = case optArg opt of
        Nothing -> "Bool"
        Just arg -> "Maybe " <> valueNameToType arg

-- | Generate the defaults value
generateDefaults :: ClapHelp -> Text
generateDefaults ClapHelp{..} =
    T.unlines $
        [ "-- | Default options - minimal flags, let the tool use its defaults"
        , "defaults :: Options"
        , "defaults = Options"
        ]
            ++ defaultDefs
            ++ [ "  }"
               ]
  where
    allOpts = dedupeByName $ filter (not . isBuiltinOpt) $ concatMap secOptions helpSections
    isBuiltinOpt opt = optLong opt `elem` [Just "help", Just "version"]

    defaultDefs = case allOpts of
        [] -> ["  { _placeholder = ()"]
        (first : rest) ->
            ("  { " <> optDefault first) : map (\o -> "  , " <> optDefault o) rest

    optDefault opt =
        let name = fieldName opt
            val = defaultValue opt
         in name <> " = " <> val

    defaultValue opt = case optArg opt of
        Nothing -> "False"
        Just _ -> "Nothing"

-- | Generate the buildArgs function using catMaybes pattern
generateBuildArgs :: ClapHelp -> Text
generateBuildArgs ClapHelp{..} =
    T.unlines $
        [ "-- | Build command-line arguments from options"
        , "buildArgs :: Options -> [Text]"
        , "buildArgs Options{..} = catMaybes"
        ]
            ++ argLines
            ++ [ "  ]"
               , "  where"
               , "    flag True  f = Just f"
               , "    flag False _ = Nothing"
               , "    opt (Just v) f = Just (f <> \"=\" <> v)"
               , "    opt Nothing  _ = Nothing"
               , "    optShow (Just v) f = Just (f <> \"=\" <> pack (show v))"
               , "    optShow Nothing  _ = Nothing"
               ]
  where
    allOpts = dedupeByName $ filter (not . isBuiltinOpt) $ concatMap secOptions helpSections
    isBuiltinOpt opt = optLong opt `elem` [Just "help", Just "version"]

    argLines = case allOpts of
        [] -> ["  ["]
        (first : rest) ->
            ("  [ " <> buildArg first) : map (\o -> "  , " <> buildArg o) rest

    buildArg opt =
        let name = fieldName opt
            flagStr = case optLong opt of
                Just long -> "--" <> long
                Nothing -> case optShort opt of
                    Just c -> "-" <> T.singleton c
                    Nothing -> "--unknown"
            argType = case optArg opt of
                Nothing -> "Bool"
                Just arg -> valueNameToType arg
         in case optArg opt of
                Nothing ->
                    "flag " <> name <> " \"" <> flagStr <> "\""
                Just _ ->
                    if argType `elem` ["Int", "Double", "FilePath"]
                        then "optShow " <> name <> " \"" <> flagStr <> "\""
                        else "opt " <> name <> " \"" <> flagStr <> "\""

-- | Generate the invocation functions
generateInvoke :: Text -> Text -> ClapHelp -> Text
generateInvoke moduleName cmdName _help =
    let funcName = T.toLower moduleName
     in T.unlines
            [ "-- | Run " <> cmdName <> " with options and additional arguments"
            , "--"
            , "-- Returns stdout. Throws on non-zero exit."
            , funcName <> " :: Options -> [Text] -> Sh Text"
            , funcName <> " opts args = run \"" <> cmdName <> "\" (buildArgs opts ++ args)"
            , ""
            , "-- | Run " <> cmdName <> ", ignoring output"
            , funcName <> "_ :: Options -> [Text] -> Sh ()"
            , funcName <> "_ opts args = run_ \"" <> cmdName <> "\" (buildArgs opts ++ args)"
            ]

-- | Compute field name from option (shared helper)
fieldName :: ClapOption -> Text
fieldName opt = case optLong opt of
    Just long ->
        let name = optionToHaskellName long
         in if isReserved name then name <> "_" else name
    Nothing -> case optShort opt of
        Just c -> "opt" <> T.singleton (toUpper c)
        Nothing -> "optUnknown"
  where
    -- Haskell keywords and common problematic names
    isReserved n =
        n
            `elem` [ "type"
                   , "class"
                   , "data"
                   , "default"
                   , "module"
                   , "where"
                   , "let"
                   , "in"
                   , "do"
                   , "case"
                   , "of"
                   , "if"
                   , "then"
                   , "else"
                   , "import"
                   , "qualified"
                   , "as"
                   , "hiding"
                   , "deriving"
                   , "instance"
                   , "newtype"
                   , "forall"
                   , "foreign"
                   , "infix"
                   , "infixl"
                   , "infixr"
                   , -- Common Prelude names that cause shadowing warnings
                     "null"
                   , "and"
                   , "or"
                   , "not"
                   , "id"
                   , "map"
                   , "filter"
                   , "head"
                   , "tail"
                   , "last"
                   , "init"
                   , "length"
                   , "reverse"
                   , "foldr"
                   , "foldl"
                   , "sum"
                   , "product"
                   , "maximum"
                   , "minimum"
                   , "concat"
                   , "print"
                   , "error"
                   ]

-- | Remove duplicate options by field name (keeps first occurrence)
dedupeByName :: [ClapOption] -> [ClapOption]
dedupeByName = go []
  where
    go seen [] = reverse seen
    go seen (opt : rest)
        | fieldName opt `elem` map fieldName seen = go seen rest
        | otherwise = go (opt : seen) rest
