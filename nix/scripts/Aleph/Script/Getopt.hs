{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Getopt
Description : Parser and code generator for GNU getopt_long CLI tools

Parse --help output from GNU coreutils and similar tools that use
getopt_long, and generate typed Haskell wrappers.

GNU getopt_long format:

@
  -a, --all                  do not ignore entries starting with .
      --author               with -l, print the author of each file
      --block-size=SIZE      with -l, scale sizes by SIZE
  -c                         with -lt: sort by ctime
      --color[=WHEN]         color the output WHEN
@
-}
module Aleph.Script.Getopt (
    -- * Types
    GetoptOption (..),
    GetoptHelp (..),

    -- * Parsing
    parseHelp,

    -- * Code generation
    generateModule,

    -- * Utilities
    optionToHaskellName,
    valueNameToType,
) where

import Control.Monad (void)
import Data.Char (isAlphaNum, isUpper, toUpper)
import Data.Text (Text)
import qualified Data.Text as T
import Data.Void
import Text.Megaparsec
import Text.Megaparsec.Char (char, digitChar, letterChar, string)

-- ============================================================================
-- Types
-- ============================================================================

-- | A CLI option/flag
data GetoptOption = GetoptOption
    { optShort :: Maybe Char
    -- ^ Short form: -x
    , optLong :: Maybe Text
    -- ^ Long form: --long-name
    , optArg :: Maybe Text
    -- ^ Argument placeholder: SIZE, FILE, etc.
    , optArgOptional :: Bool
    -- ^ Is the argument optional? [=WHEN]
    , optDesc :: Text
    -- ^ Description
    }
    deriving (Show, Eq)

-- | Full parsed help
data GetoptHelp = GetoptHelp
    { helpOptions :: [GetoptOption]
    -- ^ All options
    , helpUsage :: Maybe Text
    -- ^ Usage line if found
    }
    deriving (Show, Eq)

-- ============================================================================
-- Parser
-- ============================================================================

type Parser = Parsec Void Text

hspace :: Parser ()
hspace = void $ takeWhileP Nothing (\c -> c == ' ' || c == '\t')

hspace1 :: Parser ()
hspace1 = void $ takeWhile1P Nothing (\c -> c == ' ' || c == '\t')

restOfLine :: Parser Text
restOfLine = do
    content <- takeWhileP Nothing (/= '\n')
    void (char '\n') <|> eof
    return content

-- | Parse short option: -x
shortOpt :: Parser Char
shortOpt = char '-' *> (letterChar <|> digitChar)

-- | Parse long option: --long-name
longOpt :: Parser Text
longOpt = string "--" *> takeWhile1P Nothing isOptChar
  where
    isOptChar c = c == '-' || c == '_' || isAlphaNum c

{- | Parse argument placeholder after long option
Formats: =SIZE, [=WHEN], =FILE...
-}
argPlaceholder :: Parser (Text, Bool)
argPlaceholder =
    choice
        [ do
            -- Optional: [=WHEN]
            _ <- char '['
            _ <- optional (char '=')
            name <- takeWhile1P Nothing (\c -> c /= ']' && c /= '\n' && c /= ' ')
            _ <- char ']'
            return (name, True)
        , do
            -- Required: =SIZE
            _ <- char '='
            name <- takeWhile1P Nothing (\c -> isUpper c || c == '-' || c == '_' || c == '.')
            return (name, False)
        , do
            -- Space-separated: just an uppercase word
            hspace1
            name <- takeWhile1P Nothing (\c -> isUpper c || c == '-' || c == '_')
            -- Make sure it looks like a placeholder (all caps)
            if T.all (\c -> isUpper c || c == '-' || c == '_') name && T.length name > 0
                then return (name, False)
                else empty
        ]

{- | Parse a GNU getopt_long option line
Formats:
  -a, --all                  description
      --author               description
  -c                         description
      --block-size=SIZE      description
      --color[=WHEN]         description
-}
optionLine :: Parser GetoptOption
optionLine = do
    -- GNU style: 2-6 spaces indentation before - or --
    spaces <- takeWhile1P Nothing (== ' ')
    let indent = T.length spaces
    if indent < 2 || indent > 8
        then empty
        else return ()

    -- Parse short option (optional)
    mShort <- optional $ try $ do
        c <- shortOpt
        -- Skip ", " separator if present
        _ <- optional $ try $ string ", "
        return c

    -- Parse long option (optional)
    mLong <- optional $ try longOpt

    -- Parse argument placeholder (optional)
    mArg <- optional $ try argPlaceholder

    -- Skip to description (at least 2 spaces)
    hspace

    -- Rest is description
    desc <- restOfLine

    -- Must have at least short or long option
    case (mShort, mLong) of
        (Nothing, Nothing) -> empty
        _ ->
            return
                GetoptOption
                    { optShort = mShort
                    , optLong = mLong
                    , optArg = fst <$> mArg
                    , optArgOptional = maybe False snd mArg
                    , optDesc = T.strip desc
                    }

-- | Parse GNU getopt_long help output
parseHelp :: Text -> GetoptHelp
parseHelp input =
    let lns = T.lines input
        opts = parseLines lns
        usage = findUsage lns
     in GetoptHelp{helpOptions = opts, helpUsage = usage}
  where
    parseLines [] = []
    parseLines (line : rest) =
        case parse optionLine "" (line <> "\n") of
            Right opt -> opt : parseLines rest
            Left _ -> parseLines rest

    findUsage lns = case filter (T.isPrefixOf "Usage:") lns of
        (u : _) -> Just u
        [] -> Nothing

-- ============================================================================
-- Code Generation Utilities
-- ============================================================================

-- | Convert option long name to valid Haskell identifier
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

    ensureValidStart :: Text -> Text
    ensureValidStart t = case T.uncons t of
        Just (c, _) | c >= '0' && c <= '9' -> "opt" <> capitalize t
        _ -> t

-- | Convert value name placeholder to a Haskell type hint
valueNameToType :: Text -> Text
valueNameToType name = case T.toUpper name of
    "FILE" -> "FilePath"
    "PATH" -> "FilePath"
    "DIR" -> "FilePath"
    "DIRECTORY" -> "FilePath"
    "NUM" -> "Int"
    "NUMBER" -> "Int"
    "COUNT" -> "Int"
    "SIZE" -> "Text" -- Could be "10M", "1G", etc.
    "N" -> "Int"
    "SECONDS" -> "Int"
    "SECS" -> "Int"
    "PATTERN" -> "Text"
    "REGEX" -> "Text"
    "STRING" -> "Text"
    "NAME" -> "Text"
    "URL" -> "Text"
    "WHEN" -> "Text" -- Usually: always/auto/never
    "FORMAT" -> "Text"
    "LEVEL" -> "Text"
    "MODE" -> "Text"
    "TYPE" -> "Text"
    "STYLE" -> "Text"
    "WORD" -> "Text"
    "COMMAND" -> "Text"
    "CMD" -> "Text"
    "SHELL" -> "Text"
    "USER" -> "Text"
    "GROUP" -> "Text"
    "HOST" -> "Text"
    "PORT" -> "Int"
    _ -> "Text" -- Default to Text for unknown types

-- | Compute field name from option
fieldName :: GetoptOption -> Text
fieldName opt = case optLong opt of
    Just long ->
        let name = optionToHaskellName long
         in if isReserved name then name <> "_" else name
    Nothing -> case optShort opt of
        Just c -> "opt" <> T.singleton (toUpper c)
        Nothing -> "optUnknown"
  where
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
                   , "null"
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

-- | Compute field type from option
fieldType :: GetoptOption -> Text
fieldType opt = case optArg opt of
    Nothing -> "Bool"
    Just arg ->
        let baseType = valueNameToType arg
         in if optArgOptional opt
                then "Maybe " <> baseType -- Optional arg: Maybe (Maybe T) would be weird, just Maybe T
                else "Maybe " <> baseType

-- ============================================================================
-- Code Generation
-- ============================================================================

-- | Names exported by Aleph.Script that might conflict with command names
straylightScriptExports :: [Text]
straylightScriptExports =
    [ "ls"
    , "cd"
    , "pwd"
    , "rm"
    , "cp"
    , "mv"
    , "mkdir"
    , "find"
    , "echo"
    , "exit"
    , "run"
    , "run_"
    , "cmd"
    , "test"
    , "which"
    , "time"
    ]

-- | Generate a complete Haskell module for a CLI tool
generateModule :: Text -> Text -> GetoptHelp -> Text
generateModule moduleName cmdName help =
    let opts = filter (not . isBuiltinOpt) (helpOptions help)
        hasFilePath = any usesFilePath opts
        usesFilePath opt = case optArg opt of
            Just arg -> valueNameToType arg == "FilePath"
            Nothing -> False
        funcName = T.toLower moduleName
        -- Build list of names to hide from Aleph.Script
        -- FilePath conflicts with System.FilePath.Posix which we need for FilePath type
        hideList =
            ["FilePath" | hasFilePath]
                ++ filter (== funcName) straylightScriptExports
        importLine =
            if null hideList
                then "import Aleph.Script"
                else "import Aleph.Script hiding (" <> T.intercalate ", " hideList <> ")"
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
            , generateOptions opts
            , ""
            , generateDefaults opts
            , ""
            , generateBuildArgs opts
            , ""
            , generateInvoke moduleName cmdName
            ]
  where
    isBuiltinOpt opt = optLong opt `elem` [Just "help", Just "version"]

-- | Generate the Options record
generateOptions :: [GetoptOption] -> Text
generateOptions opts =
    T.unlines $
        [ "-- | Options record"
        , "--"
        , "-- Use 'defaults' and override fields as needed."
        , "--"
        , "data Options = Options"
        ]
            ++ fieldDefs
            ++ [ "  } deriving (Show, Eq)"
               ]
  where
    uniqueOpts = dedupeByName opts

    fieldDefs = case uniqueOpts of
        [] -> ["  { _placeholder :: ()"]
        (first : rest) ->
            ("  { " <> optField first) : map (\o -> "  , " <> optField o) rest

    optField opt =
        let name = fieldName opt
            typ = fieldType opt
            short = maybe "" (\c -> "-" <> T.singleton c <> ": ") (optShort opt)
            desc = T.take 50 (optDesc opt)
            comment = if T.null desc then "" else " -- ^ " <> short <> desc
         in name <> " :: " <> typ <> comment

-- | Generate defaults value
generateDefaults :: [GetoptOption] -> Text
generateDefaults opts =
    T.unlines $
        [ "-- | Default options"
        , "defaults :: Options"
        , "defaults = Options"
        ]
            ++ defaultDefs
            ++ [ "  }"
               ]
  where
    uniqueOpts = dedupeByName opts

    defaultDefs = case uniqueOpts of
        [] -> ["  { _placeholder = ()"]
        (first : rest) ->
            ("  { " <> optDefault first) : map (\o -> "  , " <> optDefault o) rest

    optDefault opt =
        let name = fieldName opt
            val = case optArg opt of
                Nothing -> "False"
                Just _ -> "Nothing"
         in name <> " = " <> val

-- | Generate buildArgs function
generateBuildArgs :: [GetoptOption] -> Text
generateBuildArgs opts =
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
    uniqueOpts = dedupeByName opts

    argLines = case uniqueOpts of
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

-- | Generate invocation functions
generateInvoke :: Text -> Text -> Text
generateInvoke moduleName cmdName =
    let funcName = T.toLower moduleName
     in T.unlines
            [ "-- | Run " <> cmdName <> " with options and additional arguments"
            , funcName <> " :: Options -> [Text] -> Sh Text"
            , funcName <> " opts args = run \"" <> cmdName <> "\" (buildArgs opts ++ args)"
            , ""
            , "-- | Run " <> cmdName <> ", ignoring output"
            , funcName <> "_ :: Options -> [Text] -> Sh ()"
            , funcName <> "_ opts args = run_ \"" <> cmdName <> "\" (buildArgs opts ++ args)"
            ]

-- | Remove duplicate options by field name
dedupeByName :: [GetoptOption] -> [GetoptOption]
dedupeByName = go []
  where
    go seen [] = reverse seen
    go seen (opt : rest)
        | fieldName opt `elem` map fieldName seen = go seen rest
        | otherwise = go (opt : seen) rest
