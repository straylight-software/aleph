{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- |
-- Module      : Render.Bash.Patterns
-- Description : Pattern recognition for bash constructs
--
-- Recognizes:
--   - ${VAR:-default}   parameter expansion with default
--   - ${VAR:?}          required parameter
--   - ${VAR:+alt}       alternative value
--   - config.x.y=$VAR   config assignment
--   - /nix/store/...    store paths
module Render.Bash.Patterns
  ( -- * Parameter expansion
    ParamExpansion (..),
    parseParamExpansion,

    -- * Config assignment
    ConfigAssignment (..),
    parseConfigAssignment,
    parseConfigValue,

    -- * Literals
    parseLiteral,
    isNumericLiteral,
    isBoolLiteral,
  )
where

import Data.Char (isDigit)
import Data.Text (Text)
import qualified Data.Text as T
import Render.Types

-- ============================================================================
-- Parameter Expansion
-- ============================================================================

-- | Bash parameter expansion patterns
data ParamExpansion
  = -- | ${VAR:-default} or ${VAR-default}
    DefaultValue Text (Maybe Text)
  | -- | ${VAR:=default} - assign default
    AssignDefault Text (Maybe Text)
  | -- | ${VAR:?message} or ${VAR:?} - error if unset
    ErrorIfUnset Text (Maybe Text)
  | -- | ${VAR:+alt} - use alt if set
    UseAlternate Text (Maybe Text)
  | -- | $VAR or ${VAR} - simple reference
    SimpleRef Text
  deriving (Eq, Show)

-- | Parse a parameter expansion from text
-- Handles: $VAR, ${VAR}, ${VAR:-default}, ${VAR:?}, etc.
parseParamExpansion :: Text -> Maybe ParamExpansion
parseParamExpansion t
  | "${" `T.isPrefixOf` t && "}" `T.isSuffixOf` t =
      parseExpansionBody (T.dropEnd 1 (T.drop 2 t))
  | "$" `T.isPrefixOf` t =
      Just (SimpleRef (T.drop 1 t))
  | otherwise = Nothing

parseExpansionBody :: Text -> Maybe ParamExpansion
parseExpansionBody body = case T.breakOn ":" body of
  (var, rest)
    | T.null rest -> Just (SimpleRef var)
    | ":-" `T.isPrefixOf` rest ->
        Just (DefaultValue var (nonEmpty $ T.drop 2 rest))
    | "-" `T.isPrefixOf` (T.drop 1 rest) ->
        Just (DefaultValue var (nonEmpty $ T.drop 2 rest))
    | ":=" `T.isPrefixOf` rest ->
        Just (AssignDefault var (nonEmpty $ T.drop 2 rest))
    | ":?" `T.isPrefixOf` rest ->
        Just (ErrorIfUnset var (nonEmpty $ T.drop 2 rest))
    | ":+" `T.isPrefixOf` rest ->
        Just (UseAlternate var (nonEmpty $ T.drop 2 rest))
    | otherwise -> Just (SimpleRef var)
  where
    nonEmpty x = if T.null x then Nothing else Just x

-- ============================================================================
-- Config Assignment
-- ============================================================================

-- | config.* assignment pattern
data ConfigAssignment = ConfigAssignment
  { configPath :: [Text],
    configValue :: Either Text Literal, -- Left = var reference, Right = literal
    configQuoted :: Quoted
  }
  deriving (Eq, Show)

-- | Parse config.x.y.z=$VAR or config.x.y.z="value"
parseConfigAssignment :: Text -> Maybe ConfigAssignment
parseConfigAssignment line = do
  -- Split on first =
  let (lhs, rest) = T.breakOn "=" line
  -- Check it starts with config.
  path <- T.stripPrefix "config." lhs
  -- Must have = and something after
  guard (not (T.null rest))
  let rhs = T.drop 1 rest -- drop the =
  -- Parse the path
  let pathParts = T.splitOn "." path
  -- Parse the value
  (value, quoted) <- parseConfigValue rhs
  Just
    ConfigAssignment
      { configPath = pathParts,
        configValue = value,
        configQuoted = quoted
      }
  where
    guard False = Nothing
    guard True = Just ()

parseConfigValue :: Text -> Maybe (Either Text Literal, Quoted)
parseConfigValue t
  -- Quoted brace variable: "${VAR}"
  | "\"${" `T.isPrefixOf` t && "}\"" `T.isSuffixOf` t =
      Just (Left (T.dropEnd 2 (T.drop 3 t)), Quoted)
  -- Quoted simple variable: "$VAR"
  | "\"$" `T.isPrefixOf` t && "\"" `T.isSuffixOf` t =
      Just (Left (T.dropEnd 1 (T.drop 2 t)), Quoted)
  -- Quoted literal string: "hello"
  | "\"" `T.isPrefixOf` t && "\"" `T.isSuffixOf` t =
      Just (Right (LitString (T.dropEnd 1 (T.drop 1 t))), Quoted)
  -- Unquoted brace variable: ${VAR}
  | "${" `T.isPrefixOf` t && "}" `T.isSuffixOf` t =
      Just (Left (T.dropEnd 1 (T.drop 2 t)), Unquoted)
  -- Unquoted simple variable: $VAR
  | "$" `T.isPrefixOf` t =
      Just (Left (T.drop 1 t), Unquoted)
  -- Unquoted literal
  | otherwise =
      Just (Right (parseLiteralValue t), Unquoted)

parseLiteralValue :: Text -> Literal
parseLiteralValue t
  | t == "true" = LitBool True
  | t == "false" = LitBool False
  | isNumericLiteral t = LitInt (read (T.unpack t))
  | otherwise = LitString t

-- ============================================================================
-- Literals
-- ============================================================================

-- | Parse a literal value from text
parseLiteral :: Text -> Literal
parseLiteral t
  | t == "true" = LitBool True
  | t == "false" = LitBool False
  | isNumericLiteral t = LitInt (read (T.unpack t))
  | isStorePath t = LitPath (StorePath t)
  | otherwise = LitString t

-- | Check if text is a numeric literal
-- Must have at least one digit, optional leading minus
isNumericLiteral :: Text -> Bool
isNumericLiteral t =
  not (T.null t)
    && T.all isDigitOrSign t
    && T.any isDigit t  -- Must have at least one digit
    && validMinus t
  where
    isDigitOrSign c = isDigit c || c == '-'
    -- Minus only valid at start, and only one
    validMinus s = case T.uncons s of
      Just ('-', rest) -> not (T.any (== '-') rest)
      _ -> not (T.any (== '-') s)

-- | Check if text is a boolean literal
isBoolLiteral :: Text -> Bool
isBoolLiteral t = t == "true" || t == "false"
