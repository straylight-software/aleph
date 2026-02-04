{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

-- |
-- Module      : NixCompile.Nix.Pretty
-- Description : Pretty print Nix code with type annotations
--
-- Injects type signatures as comments into the original source code.
module NixCompile.Nix.Pretty
  ( annotateSource,
  )
where

import Data.List (sortBy)
import Data.Ord (comparing)
import Data.Text (Text)
import qualified Data.Text as T
import NixCompile.Nix.Infer (Binding (..), InferResult (..))
import NixCompile.Nix.Types (prettyType)
import NixCompile.Types (Loc (..), Span (..))

-- | Annotate source code with types from inference result
annotateSource :: Text -> InferResult -> Text
annotateSource src InferResult {..} =
  let -- Collect all annotations
      bindingAnns = map mkBindingAnn irBindings
      -- Sort by position (reverse order to keep indices valid)
      anns = sortBy (flip (comparing annLoc)) bindingAnns
   in foldr applyAnn src anns

-- | Annotation to insert
data Ann = Ann
  { annLoc :: !Loc,
    annText :: !Text
  }
  deriving (Eq, Show)

-- | Create an annotation for a binding
mkBindingAnn :: Binding -> Ann
mkBindingAnn Binding {..} =
  Ann
    { annLoc = spanStart bindSpan,
      annText = "# :: " <> prettyType bindType <> "\n"
    }

-- | Apply an annotation to the source text
applyAnn :: Ann -> Text -> Text
applyAnn Ann {..} src =
  let lines_ = T.lines src
      (before, after) = splitAt (locLine annLoc - 1) lines_
      -- Insert comment before the line containing the binding
      -- Use indentation from the next line
      indent = getIndent (headSafe after)
   in T.unlines $ before ++ [indent <> annText] ++ after

-- | Get indentation of a line
getIndent :: Maybe Text -> Text
getIndent Nothing = ""
getIndent (Just t) = T.takeWhile (== ' ') t

-- | Safe head
headSafe :: [a] -> Maybe a
headSafe [] = Nothing
headSafe (x : _) = Just x
