{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

-- |
-- Module      : NixCompile.Lint.Forbidden
-- Description : Detect forbidden bash constructs
--
-- Detects constructs that are banned in nix-compile:
--   - Heredocs (<<, <<-)
--   - Here-strings (<<<)
--   - eval
--   - Backticks (`cmd`)
--
-- These are errors, not warnings. No escape hatch.
module NixCompile.Lint.Forbidden
  ( -- * Types
    Violation (..),
    ViolationType (..),

    -- * Detection
    findViolations,

    -- * Formatting
    formatViolation,
    formatViolations,
  )
where

import Data.Foldable (toList)
import Data.Text (Text)
import qualified Data.Text as T
import NixCompile.Types (Loc (..), Span (..))
import ShellCheck.AST

-- | Type of violation
data ViolationType
  = VHeredoc -- ^ << or <<-
  | VHereString -- ^ <<<
  | VEval -- ^ eval command
  | VBacktick -- ^ `cmd` syntax
  deriving (Eq, Show)

-- | A detected violation
data Violation = Violation
  { vType :: !ViolationType,
    vSpan :: !Span,
    vContext :: !Text -- ^ The offending code snippet
  }
  deriving (Eq, Show)

-- | Find all forbidden constructs in a bash AST
findViolations :: Token -> [Violation]
findViolations = go
  where
    go :: Token -> [Violation]
    go tok@(OuterToken id inner) =
      localViolations id inner ++ concatMap go (toList inner)

-- | Check a single node for violations
localViolations :: Id -> InnerToken Token -> [Violation]
localViolations id inner = case inner of
  -- Heredoc: << EOF or <<- EOF
  Inner_T_HereDoc {} ->
    [Violation VHeredoc (mkSpan id) "heredoc (<<)"]
  -- Here-string: <<< "string"
  Inner_T_HereString {} ->
    [Violation VHereString (mkSpan id) "here-string (<<<)"]
  -- Backticks: `cmd`
  Inner_T_Backticked {} ->
    [Violation VBacktick (mkSpan id) "backticks (`...`)"]
  -- Simple command: check for eval
  Inner_T_SimpleCommand _ words ->
    checkForEval id words
  _ -> []

-- | Check if a command is 'eval'
checkForEval :: Id -> [Token] -> [Violation]
checkForEval id words = case words of
  (cmdTok : _)
    | isEvalCommand cmdTok ->
        [Violation VEval (mkSpan id) "eval"]
  _ -> []

-- | Check if a token is the 'eval' command
isEvalCommand :: Token -> Bool
isEvalCommand (OuterToken _ inner) = case inner of
  Inner_T_NormalWord [OuterToken _ (Inner_T_Literal "eval")] -> True
  Inner_T_Literal "eval" -> True
  _ -> False

-- | Make a span from token ID
mkSpan :: Id -> Span
mkSpan (Id n) = Span (Loc n 0) (Loc n 0) Nothing

-- | Format a single violation for display
formatViolation :: Violation -> Text
formatViolation Violation {..} =
  let typeStr = case vType of
        VHeredoc -> "heredoc"
        VHereString -> "here-string"
        VEval -> "eval"
        VBacktick -> "backtick"
      line = locLine (spanStart vSpan)
      suggestion = case vType of
        VHeredoc ->
          T.unlines
            [ "  Use Dhall for structured output:",
              "    config=$(dhall-to-json --file config.dhall)",
              "",
              "  Or printf for simple strings:",
              "    printf 'Hello, %s\\n' \"$NAME\"",
              "",
              "  Or generate content in Nix, reference in bash:",
              "    cat ${pkgs.writeText \"msg\" ''...''}"
            ]
        VHereString ->
          T.unlines
            [ "  Use echo with pipe:",
              "    echo \"string\" | command",
              "",
              "  Or printf:",
              "    printf '%s' \"string\" | command"
            ]
        VEval ->
          T.unlines
            [ "  eval is forbidden. Refactor to avoid dynamic code execution.",
              "",
              "  If you need to set variables dynamically:",
              "    declare \"$name=$value\"",
              "",
              "  If you need to run a command from a variable:",
              "    \"$cmd\" \"$arg1\" \"$arg2\""
            ]
        VBacktick ->
          T.unlines
            [ "  Use $() instead of backticks:",
              "    result=$(command)",
              "",
              "  Not:",
              "    result=`command`"
            ]
   in T.unlines
        [ "error[RENDER-" <> code vType <> "]: " <> typeStr <> " not allowed",
          "  --> line " <> T.pack (show line),
          "",
          suggestion
        ]
  where
    code VHeredoc = "E001"
    code VHereString = "E002"
    code VEval = "E003"
    code VBacktick = "E004"

-- | Format all violations
formatViolations :: [Violation] -> Text
formatViolations [] = ""
formatViolations vs = T.intercalate "\n" (map formatViolation vs)
