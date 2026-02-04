{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- |
-- Module      : NixCompile.Nix.Layout
-- Description : Flake layout validation
--
-- Validates flake directory structure according to RFC-013.
--
-- Layout rules:
--   - No _index.nix files (graph derived from directory scan)
--   - No _main.nix files (explicit imports in flake.nix)
--   - All modules must have _class attribute
--   - Directory structure matches kind (flake/, nixos/, home/)
--   - One module per file (filename = export name)
--
-- The goal: if layout validation passes, the module graph is statically
-- extractable without Nix evaluation.
module NixCompile.Nix.Layout
  ( -- * Violations
    LayoutViolation (..),
    LayoutCode (..),
    
    -- * Validation
    findLayoutViolations,
    findLayoutViolationsInDir,
    
    -- * Queries
    expectedModuleClass,
    isIndexFile,
    isMainFile,
  )
where

import Control.Monad (forM)
import Data.Coerce (coerce)
import Data.Fix (Fix (..))
import Data.Functor.Compose (Compose (..))
import Data.List (isPrefixOf, isSuffixOf)
import Data.List.NonEmpty (NonEmpty (..))
import Data.Maybe (catMaybes, mapMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import Nix.Atoms (NAtom (..))
import Nix.Expr.Types hiding (Binding)
import qualified Nix.Expr.Types as Nix
import Nix.Expr.Types.Annotated
import Nix.Parser (parseNixFileLoc)
import qualified Nix.Utils as NixPath
import NixCompile.Types (Span (..), Loc (..))
import System.Directory (doesDirectoryExist, doesFileExist, listDirectory)
import System.FilePath ((</>), takeBaseName, takeDirectory, takeFileName, splitPath, joinPath, makeRelative, normalise)
import Text.Megaparsec.Pos (unPos)

-- ============================================================================
-- Types
-- ============================================================================

-- | Layout violation codes
data LayoutCode
  = L001  -- _index.nix file found
  | L002  -- _main.nix file found
  | L003  -- Module missing _class attribute
  | L004  -- Wrong _class for directory
  | L005  -- Invalid module location
  deriving (Show, Eq)

-- | A layout violation
data LayoutViolation = LayoutViolation
  { lvCode :: !LayoutCode
  , lvPath :: !FilePath
  , lvMessage :: !Text
  , lvSpan :: !(Maybe Span)  -- Location in file if applicable
  }
  deriving (Show, Eq)

-- ============================================================================
-- File Classification
-- ============================================================================

-- | Check if a file is an index file (banned)
isIndexFile :: FilePath -> Bool
isIndexFile path = takeFileName path == "_index.nix"

-- | Check if a file is a main file (banned)
isMainFile :: FilePath -> Bool
isMainFile path = takeFileName path == "_main.nix"

-- | Get the expected _class for a module based on its directory
-- Returns Nothing if the path isn't in a known module directory
expectedModuleClass :: FilePath -> Maybe Text
expectedModuleClass path = case getModuleKind (normalise path) of
  Just "flake" -> Just "flake"
  Just "nixos" -> Just "nixos"
  Just "home" -> Just "home"
  _ -> Nothing

-- | Extract the module kind from a path
-- e.g., "nix/modules/flake/foo.nix" -> Just "flake"
-- Uses normalised path to handle ../.. references
getModuleKind :: FilePath -> Maybe String
getModuleKind path =
  let -- Normalise the path to resolve ../.. references
      normPath = normalise path
      parts = splitPath normPath
      -- Look for pattern: .../modules/<kind>/...
      findKind [] = Nothing
      findKind [_] = Nothing
      findKind (x:y:rest)
        | "modules" `isPrefixOf` x || "modules/" `isSuffixOf` x =
            Just (dropTrailingSlash y)
        | otherwise = findKind (y:rest)
      dropTrailingSlash s = if "/" `isSuffixOf` s then init s else s
  in findKind parts

-- ============================================================================
-- Validation
-- ============================================================================

-- | Find layout violations in a single file
findLayoutViolations :: FilePath -> NExprLoc -> [LayoutViolation]
findLayoutViolations path expr = concat
  [ checkIndexFile path
  , checkMainFile path
  , checkModuleClass path expr
  ]

-- | Check for banned _index.nix
checkIndexFile :: FilePath -> [LayoutViolation]
checkIndexFile path
  | isIndexFile path = 
      [ LayoutViolation
          { lvCode = L001
          , lvPath = path
          , lvMessage = "_index.nix files are banned; module graph is derived from directory structure"
          , lvSpan = Nothing
          }
      ]
  | otherwise = []

-- | Check for banned _main.nix
checkMainFile :: FilePath -> [LayoutViolation]
checkMainFile path
  | isMainFile path =
      [ LayoutViolation
          { lvCode = L002
          , lvPath = path
          , lvMessage = "_main.nix files are banned; use explicit imports in flake.nix"
          , lvSpan = Nothing
          }
      ]
  | otherwise = []

-- | Check that modules in nix/modules/ have correct _class
checkModuleClass :: FilePath -> NExprLoc -> [LayoutViolation]
checkModuleClass path expr = case expectedModuleClass path of
  Nothing -> []  -- Not in a modules directory
  Just expectedClass ->
    case findClassAttr expr of
      Nothing ->
        [ LayoutViolation
            { lvCode = L003
            , lvPath = path
            , lvMessage = "Module missing _class attribute; expected _class = \"" <> expectedClass <> "\""
            , lvSpan = Nothing
            }
        ]
      Just (actualClass, span)
        | actualClass /= expectedClass ->
            [ LayoutViolation
                { lvCode = L004
                , lvPath = path
                , lvMessage = "Wrong _class: got \"" <> actualClass <> "\", expected \"" <> expectedClass <> "\""
                , lvSpan = Just span
                }
            ]
        | otherwise -> []

-- | Find _class attribute in a Nix expression
-- Handles both direct attr sets and function bodies
findClassAttr :: NExprLoc -> Maybe (Text, Span)
findClassAttr = go
  where
    go (Fix (Compose (AnnUnit srcSpan e))) = case e of
      -- Direct attr set: { _class = "foo"; ... }
      NSet _ bindings -> findInBindings bindings
      
      -- Function: { ... }: { _class = "foo"; ... }
      NAbs _ body -> go body
      
      -- Let: let ... in { _class = "foo"; ... }
      NLet _ body -> go body
      
      -- With: with ...; { _class = "foo"; ... }
      NWith _ body -> go body
      
      _ -> Nothing
    
    findInBindings bindings = 
      let classes = mapMaybe extractClass bindings
      in case classes of
        ((cls, sp):_) -> Just (cls, sp)
        [] -> Nothing
    
    extractClass :: Nix.Binding NExprLoc -> Maybe (Text, Span)
    extractClass = \case
      Nix.NamedVar (StaticKey name :| []) valExpr _ 
        | varNameText name == "_class" -> extractStringValue valExpr
      _ -> Nothing
    
    extractStringValue :: NExprLoc -> Maybe (Text, Span)
    extractStringValue (Fix (Compose (AnnUnit srcSpan e))) = case e of
      NStr (DoubleQuoted [Plain t]) -> Just (t, toSpan srcSpan)
      NStr (Indented _ [Plain t]) -> Just (t, toSpan srcSpan)
      _ -> Nothing
    
    varNameText :: VarName -> Text
    varNameText = coerce
    
    toSpan :: SrcSpan -> Span
    toSpan srcSpan = Span
      { spanStart = Loc (sourceLine $ getSpanBegin srcSpan) (sourceCol $ getSpanBegin srcSpan)
      , spanEnd = Loc (sourceLine $ getSpanEnd srcSpan) (sourceCol $ getSpanEnd srcSpan)
      , spanFile = Nothing
      }
    
    sourceLine (NSourcePos _ (NPos l) _) = fromIntegral (unPos l)
    sourceCol (NSourcePos _ _ (NPos c)) = fromIntegral (unPos c)

-- ============================================================================
-- Directory Scanning
-- ============================================================================

-- | Find all layout violations in a directory tree
findLayoutViolationsInDir :: FilePath -> IO [LayoutViolation]
findLayoutViolationsInDir rootDir = do
  files <- findNixFiles rootDir
  violations <- forM files $ \path -> do
    result <- parseNixFileLoc (NixPath.Path path)
    case result of
      Left _ -> pure []  -- Parse errors are handled elsewhere
      Right expr -> pure $ findLayoutViolations path expr
  pure $ concat violations

-- | Recursively find all .nix files
findNixFiles :: FilePath -> IO [FilePath]
findNixFiles dir = do
  exists <- doesDirectoryExist dir
  if not exists
    then pure []
    else do
      entries <- listDirectory dir
      paths <- forM entries $ \entry -> do
        let path = dir </> entry
        isDir <- doesDirectoryExist path
        isFile <- doesFileExist path
        if isDir
          then findNixFiles path
          else if isFile && ".nix" `isSuffixOf` entry
            then pure [path]
            else pure []
      pure $ concat paths
