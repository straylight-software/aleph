{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- |
-- Module      : NixCompile.Nix.Format
-- Description : Format Nix files with type annotations
--
-- Adds type signature comments to Nix functions and bindings.
-- The goal is to make Nix feel like a typed language.
--
-- Example output:
--
-- @
-- # mkService : { port : Int, host : String } -> Derivation
-- mkService = { port ? 8080, host ? "localhost" }:
--   pkgs.writeShellApplication { ... };
-- @
module NixCompile.Nix.Format
  ( -- * Formatting
    formatFile,
    formatExpr,
    
    -- * Annotation extraction
    extractAnnotations,
    Annotation (..),
  )
where

import Control.Monad.State.Strict
import Data.Coerce (coerce)
import Data.Fix (Fix (..))
import Data.Functor.Compose (Compose (..))
import Data.List (sortBy)
import Data.List.NonEmpty (NonEmpty (..))
import qualified Data.List.NonEmpty as NE
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Ord (comparing)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Nix.Expr.Types hiding (Binding)
import qualified Nix.Expr.Types as Nix
import Nix.Expr.Types.Annotated
import Nix.Parser (parseNixFileLoc, parseNixTextLoc)
import qualified Nix.Utils as Nix
import NixCompile.Nix.Infer (inferExpr)
import NixCompile.Nix.Types
import Text.Megaparsec.Pos (unPos)

-- ============================================================================
-- Annotations
-- ============================================================================

-- | A type annotation to insert
data Annotation = Annotation
  { annName :: !Text           -- Binding name
  , annType :: !NixType        -- Inferred type
  , annLine :: !Int            -- Line number (1-based, insert BEFORE this line)
  , annCol :: !Int             -- Column (for indentation)
  }
  deriving (Eq, Show)

-- ============================================================================
-- Extraction
-- ============================================================================

-- | Extract type annotations from a Nix expression
-- Only annotates top-level bindings (let bindings and attrset fields),
-- not nested expressions.
extractAnnotations :: NExprLoc -> [Annotation]
extractAnnotations = go True  -- Start at top level
  where
    go :: Bool -> NExprLoc -> [Annotation]
    go topLevel expr@(Fix (Compose (AnnUnit srcSpan e))) = case e of
      -- Top-level lambda: { pkgs, lib }: body
      -- Skip the lambda params, annotate the body
      NAbs _ body -> go topLevel body
      
      -- Let bindings - annotate each binding
      NLet bindings body -> 
        concatMap annotateBinding bindings ++ go False body
      
      -- Attribute set - annotate each binding (if at top level or in let)
      NSet _ bindings -> concatMap annotateBinding bindings
      
      -- Other expressions at top level - look for let/set inside
      NIf _ t f | topLevel -> go False t ++ go False f
      NWith _ body | topLevel -> go topLevel body
      NAssert _ body | topLevel -> go topLevel body
      
      _ -> []
    
    annotateBinding :: Nix.Binding NExprLoc -> [Annotation]
    annotateBinding binding = case binding of
      Nix.NamedVar (StaticKey name :| []) expr pos -> 
        let line = sourceLine pos
            col = sourceCol pos
        in case inferExpr expr of
          Right (t, _) -> 
            [ Annotation
                { annName = varNameText name
                , annType = t
                , annLine = line
                , annCol = col
                }
            ]
          Left _ -> []
      
      Nix.NamedVar _ _ _ -> []
      Nix.Inherit _ _ _ -> []
    
    sourceLine (NSourcePos _ (NPos l) _) = fromIntegral (unPos l)
    sourceCol (NSourcePos _ _ (NPos c)) = fromIntegral (unPos c)

-- | Extract text from VarName
varNameText :: VarName -> Text
varNameText = coerce

-- ============================================================================
-- Formatting
-- ============================================================================

-- | Format a Nix file with type annotations
formatFile :: FilePath -> IO (Either Text Text)
formatFile path = do
  -- Read original source
  src <- TIO.readFile path
  
  -- Parse and extract annotations
  result <- parseNixFileLoc (Nix.Path path)
  case result of
    Left doc -> pure $ Left (T.pack $ show doc)
    Right expr -> do
      let annotations = extractAnnotations expr
      pure $ Right $ insertAnnotations src annotations

-- | Format a Nix expression (from text)
formatExpr :: Text -> Either Text Text
formatExpr src = case parseNixTextLoc src of
  Left doc -> Left (T.pack $ show doc)
  Right expr -> 
    let annotations = extractAnnotations expr
    in Right $ insertAnnotations src annotations

-- | Insert annotations into source text
insertAnnotations :: Text -> [Annotation] -> Text
insertAnnotations src annotations =
  let srcLines = T.lines src
      -- Sort annotations by line (descending) so we insert from bottom up
      sortedAnns = sortBy (comparing (negate . annLine)) annotations
      -- Insert each annotation
      resultLines = foldr insertOne srcLines sortedAnns
  in T.unlines resultLines
  where
    insertOne :: Annotation -> [Text] -> [Text]
    insertOne ann lines' =
      let lineIdx = annLine ann - 1  -- Convert to 0-based
          indent = T.replicate (annCol ann - 1) " "
          comment = indent <> "# " <> annName ann <> " : " <> prettyType (annType ann)
      in if lineIdx >= 0 && lineIdx < length lines'
         then take lineIdx lines' ++ [comment] ++ drop lineIdx lines'
         else lines'
