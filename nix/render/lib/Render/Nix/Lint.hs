{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- |
-- Module      : Render.Nix.Lint
-- Description : Detect banned Nix constructs
--
-- Bans:
--   - `with expr;` - obscures scope, breaks tooling
--   - `rec { }` - enables infinite loops, complicates analysis
--
-- These are banned without escape hatch.
module Render.Nix.Lint
  ( -- * Violations
    NixViolation (..),
    ViolationType (..),
    
    -- * Detection
    findNixViolations,
    
    -- * Formatting
    formatNixViolations,
  )
where

import Data.Coerce (coerce)
import Data.Fix (Fix (..))
import Data.Functor.Compose (Compose (..))
import Data.Text (Text)
import qualified Data.Text as T
import Nix.Expr.Types
import Nix.Expr.Types.Annotated
import Render.Types (Loc (..), Span (..))
import Text.Megaparsec.Pos (unPos)

-- | Type of violation
data ViolationType
  = VWith       -- ^ `with expr;`
  | VRec        -- ^ `rec { }`
  deriving (Eq, Show)

-- | A violation found in Nix source
data NixViolation = NixViolation
  { nvType :: !ViolationType
  , nvSpan :: !Span
  , nvContext :: !Text    -- ^ Brief context (e.g., "with lib;")
  }
  deriving (Eq, Show)

-- | Find all violations in a parsed Nix expression
findNixViolations :: NExprLoc -> [NixViolation]
findNixViolations = go
  where
    go :: NExprLoc -> [NixViolation]
    go (Fix (Compose (AnnUnit srcSpan e))) = case e of
      -- with expr; body
      NWith scope body ->
        NixViolation
          { nvType = VWith
          , nvSpan = toSpan srcSpan
          , nvContext = "with " <> prettyExpr scope <> ";"
          }
        : go scope ++ go body
      
      -- rec { ... }
      NSet Recursive bindings ->
        NixViolation
          { nvType = VRec
          , nvSpan = toSpan srcSpan
          , nvContext = "rec { ... }"
          }
        : concatMap goBinding bindings
      
      -- Recurse into all other expressions
      NSet NonRecursive bindings -> concatMap goBinding bindings
      NList xs -> concatMap go xs
      NLet bindings body -> concatMap goBinding bindings ++ go body
      NIf c t f -> go c ++ go t ++ go f
      NAssert c b -> go c ++ go b
      NAbs _ b -> go b
      NApp f x -> go f ++ go x
      NSelect alt b _ -> go b ++ maybe [] go alt
      NHasAttr b _ -> go b
      NUnary _ x -> go x
      NBinary _ x y -> go x ++ go y
      _ -> []
    
    goBinding :: Binding NExprLoc -> [NixViolation]
    goBinding = \case
      NamedVar _ expr _ -> go expr
      Inherit (Just scope) _ _ -> go scope
      Inherit Nothing _ _ -> []
    
    toSpan :: SrcSpan -> Span
    toSpan srcSpan = Span
      { spanStart = Loc (sourceLine $ getSpanBegin srcSpan) (sourceCol $ getSpanBegin srcSpan)
      , spanEnd = Loc (sourceLine $ getSpanEnd srcSpan) (sourceCol $ getSpanEnd srcSpan)
      , spanFile = Nothing
      }
    
    sourceLine (NSourcePos _ (NPos l) _) = fromIntegral (unPos l)
    sourceCol (NSourcePos _ _ (NPos c)) = fromIntegral (unPos c)
    
    -- Simple expression pretty printer for context
    prettyExpr :: NExprLoc -> Text
    prettyExpr (Fix (Compose (AnnUnit _ expr))) = case expr of
      NSym name -> coerce name
      NSelect _ base _ -> prettyExpr base <> ".‥"
      _ -> "‥"

-- | Format violations for display
formatNixViolations :: [NixViolation] -> Text
formatNixViolations = T.unlines . map formatOne
  where
    formatOne v = T.unlines
      [ formatLoc (nvSpan v) <> ": " <> errorCode (nvType v)
      , "  " <> nvContext v
      , ""
      , note (nvType v)
      ]
    
    formatLoc span' = 
      T.pack (show (locLine (spanStart span'))) <> ":" <>
      T.pack (show (locCol (spanStart span')))
    
    errorCode VWith = "ALEPH-N001: `with` expression"
    errorCode VRec = "ALEPH-N002: `rec` attrset"
    
    note VWith = T.unlines
      [ "  `with` is banned because it:"
      , "    - Obscures where names come from"
      , "    - Breaks tooling (go-to-definition, autocomplete)"
      , "    - Creates shadowing hazards"
      , "    - Makes type inference unsound"
      , ""
      , "  Use `inherit (expr) name1 name2;` instead."
      ]
    note VRec = T.unlines
      [ "  `rec` is banned because it:"
      , "    - Enables infinite loops (non-termination)"
      , "    - Complicates static analysis"
      , "    - Makes evaluation order-dependent"
      , "    - Breaks referential transparency"
      , ""
      , "  Use `let` bindings or explicit function arguments instead."
      ]
