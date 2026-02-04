{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- |
-- Module      : NixCompile.Nix.Module
-- Description : Module graph construction and import following
--
-- Builds a dependency graph of Nix files by following imports.
-- This is essential for cross-file type inference.
--
-- Import patterns we handle:
--
--   import ./path.nix
--   import ./path.nix { arg = value; }
--   import ./path { inherit lib; }
--   import inputs.nixpkgs { system = "x86_64-linux"; }
--   inputs.flake-parts.lib.mkFlake { inherit inputs; } (import ./main.nix)
--
-- We track:
--   - Which files import which other files
--   - What arguments are passed to imports
--   - The type signature of each module
module NixCompile.Nix.Module
  ( -- * Module graph
    ModuleGraph (..),
    Module (..),
    Import (..),
    ParseFailure (..),
    LintFailure (..),
    LayoutFailure (..),
    
    -- * Building
    buildModuleGraph,
    buildModuleGraphFromFlake,
    
    -- * Queries
    moduleImports,
    moduleDependents,
    topologicalOrder,
    hasViolations,
    totalViolationCount,
    
    -- * Import extraction
    findImports,
  )
where

import Control.Monad (forM, foldM)
import Data.Coerce (coerce)
import Data.Fix (Fix (..))
import Data.Functor.Compose (Compose (..))
import Data.List (nub)
import Data.List.NonEmpty (NonEmpty (..))
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (catMaybes, mapMaybe)
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import Nix.Atoms (NAtom (..))
import Nix.Expr.Types hiding (Binding)
import qualified Nix.Expr.Types as Nix
import Nix.Expr.Types.Annotated
import Nix.Parser (parseNixFileLoc)
import qualified Nix.Utils as NixPath
import NixCompile.Nix.Infer (inferExpr)
import NixCompile.Nix.Layout (LayoutViolation, findLayoutViolations)
import NixCompile.Nix.Lint (NixViolation, findNixViolations)
import NixCompile.Nix.Types
import NixCompile.Types (Span (..), Loc (..))
import System.Directory (doesFileExist, canonicalizePath)
import System.FilePath ((</>), takeDirectory, normalise, makeRelative)
import Text.Megaparsec.Pos (unPos)

-- ============================================================================
-- Types
-- ============================================================================

-- | A module (Nix file) in the graph
data Module = Module
  { modPath :: !FilePath           -- Canonical path
  , modExpr :: !NExprLoc           -- Parsed AST
  , modType :: !NixType            -- Inferred type
  , modImports :: ![Import]        -- Imports from this module
  }
  deriving (Show)

-- | An import reference
data Import = Import
  { impPath :: !FilePath           -- Resolved path (canonical)
  , impRawPath :: !Text            -- Original path in source
  , impArgs :: !(Maybe NExprLoc)   -- Arguments passed (if any)
  , impSpan :: !Span               -- Source location
  }
  deriving (Show)

-- | A parse failure record
data ParseFailure = ParseFailure
  { pfPath :: !FilePath                     -- File that failed to parse
  , pfError :: !Text                        -- Error message
  }
  deriving (Show)

-- | A lint failure record (with/rec violation)
data LintFailure = LintFailure
  { lfPath :: !FilePath                     -- File with violation
  , lfViolations :: ![NixViolation]         -- Violations found
  }
  deriving (Show)

-- | A layout failure record (_index.nix, missing _class, etc.)
data LayoutFailure = LayoutFailure
  { layPath :: !FilePath                    -- File with violation
  , layViolations :: ![LayoutViolation]     -- Violations found
  }
  deriving (Show)

-- | The complete module graph
data ModuleGraph = ModuleGraph
  { mgModules :: !(Map FilePath Module)     -- All modules by path
  , mgRoot :: !FilePath                     -- Entry point (e.g., flake.nix)
  , mgOrder :: ![FilePath]                  -- Topological order (deps first)
  , mgFailures :: ![ParseFailure]           -- Files that failed to parse
  , mgLintFailures :: ![LintFailure]        -- Files with with/rec
  , mgLayoutFailures :: ![LayoutFailure]    -- Files with layout violations
  }
  deriving (Show)

-- ============================================================================
-- Building
-- ============================================================================

-- | Internal build state
data BuildState = BuildState
  { bsModules :: !(Map FilePath Module)
  , bsFailures :: ![ParseFailure]
  , bsLintFailures :: ![LintFailure]
  , bsLayoutFailures :: ![LayoutFailure]
  }

-- | Build a module graph starting from a root file
buildModuleGraph :: FilePath -> IO (Either Text ModuleGraph)
buildModuleGraph rootPath = do
  canonRoot <- canonicalizePath rootPath
  finalState <- buildModules canonRoot Set.empty (BuildState Map.empty [] [] [])
  let order = computeOrder canonRoot (bsModules finalState)
  pure $ Right $ ModuleGraph
    { mgModules = bsModules finalState
    , mgRoot = canonRoot
    , mgOrder = order
    , mgFailures = reverse (bsFailures finalState)
    , mgLintFailures = reverse (bsLintFailures finalState)
    , mgLayoutFailures = reverse (bsLayoutFailures finalState)
    }

-- | Build module graph from a flake directory
buildModuleGraphFromFlake :: FilePath -> IO (Either Text ModuleGraph)
buildModuleGraphFromFlake dir = do
  let flakePath = dir </> "flake.nix"
  exists <- doesFileExist flakePath
  if exists
    then buildModuleGraph flakePath
    else pure $ Left $ "No flake.nix found in " <> T.pack dir

-- | Recursively build modules
-- Tolerant: skips files that fail to parse, continues with others
buildModules :: FilePath -> Set FilePath -> BuildState -> IO BuildState
buildModules path visited state
  | path `Set.member` visited = pure state  -- Already processed
  | otherwise = do
      exists <- doesFileExist path
      if not exists
        then pure state  -- Skip non-existent (might be external input)
        else do
          result <- parseNixFileLoc (NixPath.Path path)
          case result of
            Left doc -> do
              -- Record the parse failure
              let failure = ParseFailure path (T.pack (show doc))
              let state' = state { bsFailures = failure : bsFailures state }
              pure state'
            Right expr -> do
              let imports = findImports (takeDirectory path) expr
              let modType = case inferExpr expr of
                    Right (t, _) -> t
                    Left _ -> TAny
              
              -- Check for with/rec violations
              let lintViolations = findNixViolations expr
              
              -- Check for layout violations
              let layoutViolations = findLayoutViolations path expr
              
              let mod = Module
                    { modPath = path
                    , modExpr = expr
                    , modType = modType
                    , modImports = imports
                    }
              
              let state' = state { bsModules = Map.insert path mod (bsModules state) }
              
              -- Record lint failures if any
              let state'' = if null lintViolations
                    then state'
                    else state' { bsLintFailures = LintFailure path lintViolations : bsLintFailures state' }
              
              -- Record layout failures if any
              let state''' = if null layoutViolations
                    then state''
                    else state'' { bsLayoutFailures = LayoutFailure path layoutViolations : bsLayoutFailures state'' }
              
              let visited' = Set.insert path visited
              
              -- Recursively process imports
              foldM (processImport visited') state''' imports

-- | Process a single import
processImport :: Set FilePath -> BuildState -> Import -> IO BuildState
processImport visited state imp = do
  exists <- doesFileExist (impPath imp)
  if exists
    then do
      -- Canonicalize the path to resolve ../.. sequences
      canonPath <- canonicalizePath (impPath imp)
      buildModules canonPath visited state
    else pure state  -- Skip non-existent imports

-- ============================================================================
-- Import Finding
-- ============================================================================

-- | Find all imports in an expression
findImports :: FilePath -> NExprLoc -> [Import]
findImports baseDir = go
  where
    go :: NExprLoc -> [Import]
    go expr@(Fix (Compose (AnnUnit srcSpan e))) = case e of
      -- import ./path  or  (import ./path) { args }
      NApp func arg -> 
        -- First check if func is directly the import builtin
        case isImportBuiltin func of
          Just () -> 
            -- func is `import`, arg is the path
            makeImport baseDir (extractPath arg) Nothing srcSpan
          Nothing ->
            -- func might be (import ./path) and arg is the args
            case unwrapImport func of
              Just (rawPath, Nothing) ->
                makeImport baseDir rawPath (Just arg) srcSpan ++ go arg
              Just (rawPath, Just inner) ->
                makeImport baseDir rawPath (Just arg) srcSpan ++ go inner ++ go arg
              Nothing -> go func ++ go arg
      
      -- Recurse into all expressions
      NLet bindings body -> 
        concatMap goBinding bindings ++ go body
      NSet _ bindings -> concatMap goBinding bindings
      NIf c t f -> go c ++ go t ++ go f
      NWith s b -> go s ++ go b
      NAssert c b -> go c ++ go b
      NAbs _ b -> go b
      NList xs -> concatMap go xs
      NSelect _ b _ -> go b
      NBinary _ l r -> go l ++ go r
      NUnary _ x -> go x
      _ -> []
    
    goBinding :: Nix.Binding NExprLoc -> [Import]
    goBinding = \case
      Nix.NamedVar _ expr _ -> go expr
      Nix.Inherit (Just scope) _ _ -> go scope
      Nix.Inherit Nothing _ _ -> []
    
    -- Try to unwrap an import expression, returning the path and any remaining args
    -- Returns: Just (path, maybeRemainingExpr) if this is an import
    unwrapImport :: NExprLoc -> Maybe (Text, Maybe NExprLoc)
    unwrapImport (Fix (Compose (AnnUnit _ e))) = case e of
      -- import path  or  import path { args }
      NApp func pathExpr -> case func of
        -- Direct: import ./path
        Fix (Compose (AnnUnit _ (NSym name))) 
          | varNameText name == "import" -> Just (extractPath pathExpr, Nothing)
        -- Curried: (import ./path) { args } - pathExpr is the args, recurse into func
        _ -> case unwrapImport func of
          Just (path, Nothing) | not (T.null path) -> Just (path, Just pathExpr)
          _ -> case isImportBuiltin func of
            Just () -> Just (extractPath pathExpr, Nothing)
            Nothing -> Nothing
      
      _ -> Nothing
    
    -- Check if expression is `import` or `builtins.import`
    isImportBuiltin :: NExprLoc -> Maybe ()
    isImportBuiltin (Fix (Compose (AnnUnit _ e))) = case e of
      NSym name | varNameText name == "import" -> Just ()
      NSelect _ _ (attr :| rest) 
        | varNameText (keyName (last (attr:rest))) == "import" -> Just ()
      _ -> Nothing
    
    -- Extract path from path expression
    extractPath :: NExprLoc -> Text
    extractPath (Fix (Compose (AnnUnit _ e))) = case e of
      NLiteralPath (NixPath.Path p) -> T.pack p
      NStr (DoubleQuoted [Plain t]) -> t
      NStr (Indented _ [Plain t]) -> t
      _ -> ""
    
    keyName (StaticKey k) = k
    keyName (DynamicKey _) = VarName ""
    
    varNameText :: VarName -> Text
    varNameText = coerce
    
    makeImport :: FilePath -> Text -> Maybe NExprLoc -> SrcSpan -> [Import]
    makeImport base rawPath args srcSpan
      | T.null rawPath = []
      | otherwise = 
          let resolved = resolveImportPath base (T.unpack rawPath)
          in [ Import
                { impPath = resolved
                , impRawPath = rawPath
                , impArgs = args
                , impSpan = toSpan srcSpan
                }
             ]
    
    toSpan :: SrcSpan -> Span
    toSpan srcSpan = Span
      { spanStart = Loc (sourceLine $ getSpanBegin srcSpan) (sourceCol $ getSpanBegin srcSpan)
      , spanEnd = Loc (sourceLine $ getSpanEnd srcSpan) (sourceCol $ getSpanEnd srcSpan)
      , spanFile = Nothing
      }
    
    sourceLine (NSourcePos _ (NPos l) _) = fromIntegral (unPos l)
    sourceCol (NSourcePos _ _ (NPos c)) = fromIntegral (unPos c)

-- | Resolve an import path relative to a base directory
resolveImportPath :: FilePath -> FilePath -> FilePath
resolveImportPath baseDir path = case path of
  '.':_ -> normalise (baseDir </> path)
  '/':_ -> path
  _     -> path  -- External input, keep as-is

-- ============================================================================
-- Queries
-- ============================================================================

-- | Get all imports for a module
moduleImports :: ModuleGraph -> FilePath -> [Import]
moduleImports mg path = maybe [] modImports (Map.lookup path (mgModules mg))

-- | Get all modules that import a given module
moduleDependents :: ModuleGraph -> FilePath -> [FilePath]
moduleDependents mg path = 
  [ modPath m 
  | m <- Map.elems (mgModules mg)
  , any (\i -> impPath i == path) (modImports m)
  ]

-- | Get topological order (dependencies first)
topologicalOrder :: ModuleGraph -> [FilePath]
topologicalOrder = mgOrder

-- | Check if the module graph has any violations
hasViolations :: ModuleGraph -> Bool
hasViolations mg = 
  not (null (mgFailures mg)) ||
  not (null (mgLintFailures mg)) ||
  not (null (mgLayoutFailures mg))

-- | Count total violations across all categories
totalViolationCount :: ModuleGraph -> Int
totalViolationCount mg =
  length (mgFailures mg) +  -- Parse failures
  sum (map (length . lfViolations) (mgLintFailures mg)) +  -- Lint violations
  sum (map (length . layViolations) (mgLayoutFailures mg))  -- Layout violations

-- | Compute topological order using DFS
computeOrder :: FilePath -> Map FilePath Module -> [FilePath]
computeOrder root modules = reverse $ snd $ dfs Set.empty [] root
  where
    dfs :: Set FilePath -> [FilePath] -> FilePath -> (Set FilePath, [FilePath])
    dfs visited order path
      | path `Set.member` visited = (visited, order)
      | otherwise = 
          case Map.lookup path modules of
            Nothing -> (visited, order)
            Just m ->
              let visited' = Set.insert path visited
                  (visited'', order') = foldl go (visited', order) (map impPath (modImports m))
              in (visited'', path : order')
    
    go (v, o) p = dfs v o p
