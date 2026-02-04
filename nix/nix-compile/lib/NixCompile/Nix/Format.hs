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
  )
where

import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Nix.Expr.Types.Annotated (NExprLoc)
import Nix.Parser (parseNixFileLoc, parseNixTextLoc)
import qualified Nix.Utils as Nix
import NixCompile.Nix.Infer (inferExpr, InferResult(..))
import NixCompile.Nix.Pretty (annotateSource)

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
    Right expr -> pure $ formatExpr' src expr

-- | Format a Nix expression (from text)
formatExpr :: Text -> Either Text Text
formatExpr src = case parseNixTextLoc src of
  Left doc -> Left (T.pack $ show doc)
  Right expr -> formatExpr' src expr

-- | Internal formatter using pre-parsed expression
formatExpr' :: Text -> NExprLoc -> Either Text Text
formatExpr' src expr = 
  case inferExpr expr of
    Left err -> Left err
    Right (t, bindings) -> 
      let res = InferResult bindings [] -- We don't track top-level functions separately here
      in Right $ annotateSource src res
