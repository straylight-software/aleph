{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

-- |
-- Module      : Render.Bash.Facts
-- Description : Extract facts from bash AST
--
-- Walks the ShellCheck AST and extracts facts about:
--   - Variable assignments and defaults
--   - Config.* assignments
--   - Command invocations
--   - Store path usage
module Render.Bash.Facts
  ( extractFacts,
  )
where

import Data.Foldable (toList)
import Data.Text (Text)
import qualified Data.Text as T
import Render.Bash.Patterns
import Render.Types hiding (RT.Quoted, RT.Unquoted)
import qualified Render.Types as RT
import ShellCheck.AST hiding (RT.Quoted, RT.Unquoted)

-- | Extract all facts from a bash AST
extractFacts :: Token -> [Fact]
extractFacts = go
  where
    go (OuterToken id inner) = localFacts id inner ++ concatMap go (toList inner)

-- | Extract facts from a single token (non-recursive)
localFacts :: Id -> InnerToken Token -> [Fact]
localFacts id inner = case inner of
  -- Assignment: VAR=value or VAR="${VAR:-default}"
  Inner_T_Assignment _ name _ value ->
    assignmentFacts (mkSpan id) (T.pack name) value
  -- Simple command: check for config.* or commands
  Inner_T_SimpleCommand assigns words ->
    commandFacts (mkSpan id) words
  _ -> []

-- | Facts from an assignment
assignmentFacts :: Span -> Text -> Token -> [Fact]
assignmentFacts span name value =
  case extractParamExpansion value of
    Just (DefaultValue var (Just def)) ->
      [DefaultIs name (parseLiteral def) span]
    Just (DefaultValue _var Nothing) ->
      [Required name span] -- No default means required
    Just (ErrorIfUnset _var _) ->
      [Required name span]
    Just (SimpleRef var) ->
      [AssignFrom name var span]
    Nothing ->
      case extractLiteral value of
        Just lit -> [AssignLit name lit span]
        Nothing -> []

-- | Facts from a command
commandFacts :: Span -> [Token] -> [Fact]
commandFacts span words = case words of
  [] -> []
  (cmdTok : args) ->
    let cmdText = tokenToText cmdTok
     in if "config." `T.isPrefixOf` cmdText
          then configFactsFromToken span cmdTok
          else cmdInvocationFacts span cmdText args

-- | Facts from config.* assignment (AST-aware)
-- We need to check the AST structure to determine if the value was quoted
configFactsFromToken :: Span -> Token -> [Fact]
configFactsFromToken span tok@(OuterToken _ inner) =
  case inner of
    -- NormalWord contains the config assignment parts
    Inner_T_NormalWord parts -> configFactsFromParts span parts
    _ -> configFacts span (tokenToText tok)

-- | Extract config facts from NormalWord parts
-- The structure is: "config.x.y=" followed by the value part(s)
configFactsFromParts :: Span -> [Token] -> [Fact]
configFactsFromParts span parts =
  let text = T.concat (map tokenToText parts)
      (lhs, rest) = T.breakOn "=" text
   in case T.stripPrefix "config." lhs of
        Just pathText | not (T.null rest) ->
          let pathParts = T.splitOn "." pathText
              -- Find the value token(s) after the "=" in the literal
              (valueToks, quoted) = findValueTokens parts
              varOrLit = extractConfigValue valueToks quoted
           in case varOrLit of
                Just (Left var) -> [ConfigAssign pathParts var quoted span]
                Just (Right lit) -> [ConfigLit pathParts lit span]
                Nothing -> []
        _ -> []

-- | Find value tokens and determine if quoted
-- Returns the tokens representing the value and whether it was quoted
findValueTokens :: [Token] -> ([Token], RT.Quoted)
findValueTokens parts = go parts False
  where
    go [] _ = ([], RT.Unquoted)
    go (t@(OuterToken _ inner) : rest) seenEq = case inner of
      Inner_T_Literal s | "=" `T.isSuffixOf` (T.pack s) ->
        -- We've hit the "=" in "config.x.y=", value starts next
        go rest True
      Inner_T_DoubleQuoted _ | seenEq ->
        -- Double-quoted value
        ([t], RT.Quoted)
      _ | seenEq ->
        -- RT.Unquoted value (could be literal or variable)
        ([t], RT.Unquoted)
      _ ->
        -- Still in the path part
        go rest seenEq

-- | Extract variable name or literal from value tokens
extractConfigValue :: [Token] -> RT.Quoted -> Maybe (Either Text Literal)
extractConfigValue [] _ = Nothing
extractConfigValue (tok:_) quoted =
  let text = tokenToText tok
   in case extractSimpleVar text of
        Just var -> Just (Left var)
        Nothing -> Just (Right (parseLiteral text))

-- | Extract simple variable reference: ${VAR} or HOST -> Just "VAR"/"HOST"
extractSimpleVar :: Text -> Maybe Text
extractSimpleVar t
  | "${" `T.isPrefixOf` t && "}" `T.isSuffixOf` t = Just (T.dropEnd 1 (T.drop 2 t))
  | T.all isVarChar t && not (T.null t) && not (isNumericLiteral t) = Just t
  | otherwise = Nothing
  where
    isVarChar c = c == '_' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')

-- | Facts from config.* assignment (text-based fallback)
configFacts :: Span -> Text -> [Fact]
configFacts span text =
  case parseConfigAssignment text of
    Just ConfigAssignment {..} ->
      case configValue of
        Left var -> [ConfigAssign configPath var configQuoted span]
        Right lit -> [ConfigLit configPath lit span]
    Nothing -> []

-- | Facts from command invocation
cmdInvocationFacts :: Span -> Text -> [Token] -> [Fact]
cmdInvocationFacts span cmd args = pathFact ++ argFacts
  where
    -- What kind of command is this?
    pathFact
      | isStorePath cmd = [UsesStorePath (StorePath cmd) span]
      | "$" `T.isPrefixOf` cmd = [DynamicCommand (T.drop 1 cmd) span]
      | otherwise = [BareCommand cmd span]

    -- Extract the command name (strip store path prefix if present)
    cmdName = extractCmdName cmd

    -- Parse arguments looking for flag-value pairs with variables
    argFacts = extractArgFacts cmdName args

-- | Extract command name from path
-- /nix/store/xxx-curl/bin/curl -> curl
extractCmdName :: Text -> Text
extractCmdName path
  | isStorePath path = case reverse (T.splitOn "/" path) of
      (name : _) | not (T.null name) -> name
      _ -> path
  | otherwise = path

-- | Extract CmdArg facts from argument list
-- Looks for patterns like: --timeout $VAR or --timeout "$VAR"
extractArgFacts :: Text -> [Token] -> [Fact]
extractArgFacts cmd tokens = go tokens
  where
    go [] = []
    go [_] = [] -- need at least flag + value
    go (flagTok : valueTok : rest) =
      let flagText = tokenToText flagTok
          valueText = tokenToText valueTok
       in case extractVarRef valueText of
            Just varName
              | isFlag flagText ->
                  CmdArg cmd flagText varName (mkSpan (tokId flagTok)) : go rest
            _ -> go (valueTok : rest)

    isFlag t = "-" `T.isPrefixOf` t

    -- Extract variable reference: "$VAR" -> Just "VAR", ${VAR} -> Just "VAR"
    extractVarRef t
      -- "${VAR}" quoted
      | "\"${" `T.isPrefixOf` t && "}\"" `T.isSuffixOf` t =
          Just (T.dropEnd 2 (T.drop 3 t))
      -- "$VAR" quoted
      | "\"$" `T.isPrefixOf` t && "\"" `T.isSuffixOf` t =
          Just (T.dropEnd 1 (T.drop 2 t))
      -- ${VAR} unquoted
      | "${" `T.isPrefixOf` t && "}" `T.isSuffixOf` t =
          Just (T.dropEnd 1 (T.drop 2 t))
      -- $VAR unquoted (not $(...))
      | "$" `T.isPrefixOf` t && not ("$(" `T.isPrefixOf` t) =
          Just (T.drop 1 t)
      | otherwise = Nothing

    tokId (OuterToken id _) = id

-- | Try to extract parameter expansion from a token
extractParamExpansion :: Token -> Maybe ParamExpansion
extractParamExpansion tok =
  parseParamExpansion (tokenToText tok)

-- | Try to extract a literal from a token
extractLiteral :: Token -> Maybe Literal
extractLiteral tok =
  let t = tokenToText tok
   in if T.null t then Nothing else Just (parseLiteral t)

-- | Convert token to text (simplified)
tokenToText :: Token -> Text
tokenToText (OuterToken _ inner) = innerToText inner

innerToText :: InnerToken Token -> Text
innerToText = \case
  Inner_T_Literal s -> T.pack s
  Inner_T_SingleQuoted s -> T.pack s
  Inner_T_Glob s -> T.pack s
  Inner_T_NormalWord parts -> T.concat (map tokenToText parts)
  Inner_T_DoubleQuoted parts -> T.concat (map tokenToText parts)
  Inner_T_DollarBraced _ t -> "${" <> tokenToText t <> "}"
  Inner_T_DollarSingleQuoted s -> T.pack s
  Inner_T_BraceExpansion parts -> T.concat (map tokenToText parts)
  _ -> ""

-- | Make a span from a token ID
mkSpan :: Id -> Span
mkSpan (Id n) = Span (Loc n 0) (Loc n 0) Nothing
