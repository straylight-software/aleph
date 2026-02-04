{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- |
-- Module      : NixCompile.Nix.Flake
-- Description : Parse and type-check Nix flakes
--
-- Understands the flake schema and provides typed access to:
--   - inputs (nixpkgs, self, etc.)
--   - outputs (packages, devShells, checks, etc.)
--   - the flake function signature
module NixCompile.Nix.Flake
  ( -- * Flake types
    Flake (..),
    FlakeInput (..),
    FlakeOutputs (..),
    OutputEntry (..),
    
    -- * Parsing
    parseFlake,
    parseFlakeDir,
    
    -- * Type inference
    inferFlake,
    FlakeTypes (..),
    
    -- * Schema
    flakeOutputSchema,
  )
where

import Control.Monad (forM)
import Data.Coerce (coerce)
import Data.Fix (Fix (..))
import Data.Functor.Compose (Compose (..))
import Data.List.NonEmpty (NonEmpty (..))
import qualified Data.List.NonEmpty as NE
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (catMaybes, mapMaybe)
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Nix.Atoms (NAtom (..))
import Nix.Expr.Types hiding (Binding)
import qualified Nix.Expr.Types as Nix
import Nix.Expr.Types.Annotated
import Nix.Parser (parseNixFileLoc)
import qualified Nix.Utils as Nix
import NixCompile.Nix.Infer (inferExpr, TypeEnv, builtinEnv)
import NixCompile.Nix.Types
import NixCompile.Types (Span (..))
import System.Directory (doesFileExist)
import System.FilePath ((</>))
import Text.Megaparsec.Pos (unPos)

-- ============================================================================
-- Flake types
-- ============================================================================

-- | A parsed flake
data Flake = Flake
  { flakeDescription :: !(Maybe Text)
  , flakeInputs :: !(Map Text FlakeInput)
  , flakeOutputs :: !FlakeOutputs
  , flakePath :: !FilePath
  }
  deriving (Eq, Show)

-- | A flake input
data FlakeInput = FlakeInput
  { inputUrl :: !(Maybe Text)      -- "github:NixOS/nixpkgs"
  , inputFlake :: !Bool            -- true by default
  , inputFollows :: !(Maybe Text)  -- "nixpkgs"
  }
  deriving (Eq, Show)

-- | Flake outputs
data FlakeOutputs = FlakeOutputs
  { outPackages :: !(Map Text (Map Text OutputEntry))      -- system -> name -> entry
  , outDevShells :: !(Map Text (Map Text OutputEntry))
  , outChecks :: !(Map Text (Map Text OutputEntry))
  , outApps :: !(Map Text (Map Text OutputEntry))
  , outOverlays :: !(Map Text OutputEntry)
  , outNixosModules :: !(Map Text OutputEntry)
  , outNixosConfigurations :: !(Map Text OutputEntry)
  , outLib :: !(Maybe NExprLoc)
  , outOther :: !(Map Text NExprLoc)
  }
  deriving (Show)

instance Eq FlakeOutputs where
  a == b = outPackages a == outPackages b

-- | An output entry
data OutputEntry = OutputEntry
  { entryName :: !Text
  , entryExpr :: !NExprLoc
  , entryType :: !NixType
  }
  deriving (Show)

instance Eq OutputEntry where
  a == b = entryName a == entryName b && entryType a == entryType b

-- ============================================================================
-- Parsing
-- ============================================================================

-- | Parse a flake.nix file
parseFlake :: FilePath -> IO (Either Text Flake)
parseFlake path = do
  result <- parseNixFileLoc (Nix.Path path)
  case result of
    Left doc -> pure $ Left (T.pack $ show doc)
    Right expr -> pure $ extractFlake path expr

-- | Parse a flake from a directory (looks for flake.nix)
parseFlakeDir :: FilePath -> IO (Either Text Flake)
parseFlakeDir dir = do
  let flakePath = dir </> "flake.nix"
  exists <- doesFileExist flakePath
  if exists
    then parseFlake flakePath
    else pure $ Left $ "No flake.nix found in " <> T.pack dir

-- | Extract flake structure from AST
extractFlake :: FilePath -> NExprLoc -> Either Text Flake
extractFlake path expr = do
  -- Flake should be an attrset at top level
  case unwrapExpr expr of
    NSet _ bindings -> do
      let desc = extractDescription bindings
      let inputs = extractInputs bindings
      let outputs = extractOutputs bindings
      Right $ Flake
        { flakeDescription = desc
        , flakeInputs = inputs
        , flakeOutputs = outputs
        , flakePath = path
        }
    _ -> Left "flake.nix must be an attribute set"

-- | Unwrap expression to get the inner NExprF
unwrapExpr :: NExprLoc -> NExprF NExprLoc
unwrapExpr (Fix (Compose (AnnUnit _ e))) = e

-- | Extract description field
extractDescription :: [Nix.Binding NExprLoc] -> Maybe Text
extractDescription = foldr check Nothing
  where
    check (Nix.NamedVar (StaticKey name :| []) expr _) acc
      | varNameText name == "description" = extractStringLit expr
      | otherwise = acc
    check _ acc = acc

-- | Extract inputs
extractInputs :: [Nix.Binding NExprLoc] -> Map Text FlakeInput
extractInputs bindings = case findBinding "inputs" bindings of
  Just expr -> case unwrapExpr expr of
    NSet _ inputBindings -> Map.fromList $ mapMaybe parseInput inputBindings
    _ -> Map.empty
  Nothing -> Map.empty
  where
    parseInput :: Nix.Binding NExprLoc -> Maybe (Text, FlakeInput)
    parseInput (Nix.NamedVar (StaticKey name :| []) expr _) =
      Just (varNameText name, parseInputExpr expr)
    parseInput _ = Nothing
    
    parseInputExpr :: NExprLoc -> FlakeInput
    parseInputExpr expr = case unwrapExpr expr of
      -- Simple: inputs.foo = { url = "..."; };
      NSet _ bs -> FlakeInput
        { inputUrl = findBinding "url" bs >>= extractStringLit
        , inputFlake = maybe True id (findBinding "flake" bs >>= extractBoolLit)
        , inputFollows = findBinding "follows" bs >>= extractStringLit
        }
      -- String shorthand: inputs.foo = "github:...";
      NStr _ -> FlakeInput
        { inputUrl = extractStringLit expr
        , inputFlake = True
        , inputFollows = Nothing
        }
      _ -> FlakeInput Nothing True Nothing

-- | Extract outputs
extractOutputs :: [Nix.Binding NExprLoc] -> FlakeOutputs
extractOutputs bindings = case findBinding "outputs" bindings of
  Just outputsExpr -> parseOutputsExpr outputsExpr
  Nothing -> emptyOutputs
  where
    emptyOutputs = FlakeOutputs Map.empty Map.empty Map.empty Map.empty 
                                Map.empty Map.empty Map.empty Nothing Map.empty

-- | Parse the outputs expression (usually a function)
parseOutputsExpr :: NExprLoc -> FlakeOutputs
parseOutputsExpr expr = case unwrapExpr expr of
  -- outputs = { self, nixpkgs, ... }: { ... }
  NAbs _ body -> parseOutputsBody body
  -- outputs = { ... } (already evaluated?)
  NSet _ bindings -> parseOutputsBindings bindings
  _ -> FlakeOutputs Map.empty Map.empty Map.empty Map.empty 
                    Map.empty Map.empty Map.empty Nothing Map.empty

-- | Parse the body of the outputs function
parseOutputsBody :: NExprLoc -> FlakeOutputs
parseOutputsBody expr = case unwrapExpr expr of
  NSet _ bindings -> parseOutputsBindings bindings
  NLet _ body -> parseOutputsBody body
  NWith _ body -> parseOutputsBody body
  _ -> FlakeOutputs Map.empty Map.empty Map.empty Map.empty 
                    Map.empty Map.empty Map.empty Nothing Map.empty

-- | Parse output bindings
parseOutputsBindings :: [Nix.Binding NExprLoc] -> FlakeOutputs
parseOutputsBindings bindings = FlakeOutputs
  { outPackages = parseSystemMap "packages"
  , outDevShells = parseSystemMap "devShells"
  , outChecks = parseSystemMap "checks"
  , outApps = parseSystemMap "apps"
  , outOverlays = parseSimpleMap "overlays"
  , outNixosModules = parseSimpleMap "nixosModules"
  , outNixosConfigurations = parseSimpleMap "nixosConfigurations"
  , outLib = findBinding "lib" bindings
  , outOther = Map.empty -- TODO
  }
  where
    parseSystemMap :: Text -> Map Text (Map Text OutputEntry)
    parseSystemMap name = case findBinding name bindings of
      Just expr -> case unwrapExpr expr of
        NSet _ systemBindings -> 
          Map.fromList $ mapMaybe (parseSystemBinding name) systemBindings
        _ -> Map.empty
      Nothing -> Map.empty
    
    parseSystemBinding :: Text -> Nix.Binding NExprLoc -> Maybe (Text, Map Text OutputEntry)
    parseSystemBinding category (Nix.NamedVar (StaticKey system :| []) expr _) =
      case unwrapExpr expr of
        NSet _ pkgBindings ->
          Just (varNameText system, Map.fromList $ mapMaybe (parseEntry category) pkgBindings)
        _ -> Nothing
    parseSystemBinding _ _ = Nothing
    
    parseSimpleMap :: Text -> Map Text OutputEntry
    parseSimpleMap name = case findBinding name bindings of
      Just expr -> case unwrapExpr expr of
        NSet _ entryBindings ->
          Map.fromList $ mapMaybe (parseEntry name) entryBindings
        _ -> Map.empty
      Nothing -> Map.empty
    
    parseEntry :: Text -> Nix.Binding NExprLoc -> Maybe (Text, OutputEntry)
    parseEntry category (Nix.NamedVar (StaticKey name :| []) expr _) =
      let entryType = inferOutputType category
          entry = OutputEntry (varNameText name) expr entryType
      in Just (varNameText name, entry)
    parseEntry _ _ = Nothing

-- | Infer the expected type for an output category
inferOutputType :: Text -> NixType
inferOutputType = \case
  "packages" -> TDerivation
  "devShells" -> TDerivation
  "checks" -> TDerivation
  "apps" -> TAttrs $ Map.fromList [("type", (TString, False)), ("program", (TString, False))]
  "overlays" -> TFun (TAttrsOpen Map.empty) (TFun (TAttrsOpen Map.empty) (TAttrsOpen Map.empty))
  "nixosModules" -> TAttrsOpen Map.empty  -- Module structure
  "nixosConfigurations" -> TAttrsOpen Map.empty  -- NixOS config
  "lib" -> TAttrsOpen Map.empty
  _ -> TAny

-- ============================================================================
-- Type inference
-- ============================================================================

-- | Inferred types for a flake
data FlakeTypes = FlakeTypes
  { ftOutputsType :: !NixType
  , ftPackageTypes :: !(Map Text (Map Text NixType))
  , ftLibTypes :: !(Maybe (Map Text NixType))
  }
  deriving (Eq, Show)

-- | Infer types for a flake
inferFlake :: Flake -> FlakeTypes
inferFlake flake = FlakeTypes
  { ftOutputsType = flakeOutputsType
  , ftPackageTypes = Map.map (Map.map entryType) (outPackages (flakeOutputs flake))
  , ftLibTypes = Nothing  -- TODO: infer lib types
  }
  where
    flakeOutputsType = TFun flakeInputsType outputsType
    
    flakeInputsType = TAttrs $ Map.fromList
      [ ("self", (TAttrsOpen Map.empty, False))
      ] `Map.union` Map.map (const $ (TAttrsOpen Map.empty, False)) (flakeInputs flake)
    
    outputsType = TAttrs $ Map.fromList $ catMaybes
      [ if Map.null (outPackages o) then Nothing else Just ("packages", (systemMapType TDerivation, False))
      , if Map.null (outDevShells o) then Nothing else Just ("devShells", (systemMapType TDerivation, False))
      , if Map.null (outChecks o) then Nothing else Just ("checks", (systemMapType TDerivation, False))
      , if Map.null (outApps o) then Nothing else Just ("apps", (systemMapType appType, False))
      , if Map.null (outOverlays o) then Nothing else Just ("overlays", (TAttrsOpen Map.empty, False))
      ]
    
    o = flakeOutputs flake
    
    systemMapType t = TAttrs $ Map.fromList
      [ ("x86_64-linux", (TAttrsOpen (Map.singleton "_" (t, False)), False))
      , ("aarch64-linux", (TAttrsOpen (Map.singleton "_" (t, False)), False))
      , ("x86_64-darwin", (TAttrsOpen (Map.singleton "_" (t, False)), False))
      , ("aarch64-darwin", (TAttrsOpen (Map.singleton "_" (t, False)), False))
      ]
    
    appType = TAttrs $ Map.fromList [("type", (TString, False)), ("program", (TString, False))]

-- ============================================================================
-- Schema
-- ============================================================================

-- | The expected flake output schema
flakeOutputSchema :: NixType
flakeOutputSchema = TAttrs $ Map.fromList
  [ ("packages", (systemMapType TDerivation, False))
  , ("devShells", (systemMapType TDerivation, False))
  , ("checks", (systemMapType TDerivation, False))
  , ("apps", (systemMapType appType, False))
  , ("overlays", (TAttrsOpen (Map.singleton "_" (overlayType, False)), False))
  , ("nixosModules", (TAttrsOpen Map.empty, False))
  , ("nixosConfigurations", (TAttrsOpen Map.empty, False))
  , ("lib", (TAttrsOpen Map.empty, False))
  , ("formatter", (systemMapType TDerivation, False))
  , ("templates", (TAttrsOpen (Map.singleton "_" (templateType, False)), False))
  ]
  where
    systemMapType t = TAttrsOpen (Map.singleton "_" ((TAttrsOpen (Map.singleton "_" (t, False))), False))
    appType = TAttrs $ Map.fromList [("type", (TString, False)), ("program", (TString, False))]
    overlayType = TFun (TAttrsOpen Map.empty) (TFun (TAttrsOpen Map.empty) (TAttrsOpen Map.empty))
    templateType = TAttrs $ Map.fromList [("description", (TString, False)), ("path", (TPath, False))]

-- ============================================================================
-- Helpers
-- ============================================================================

-- | Find a binding by name
findBinding :: Text -> [Nix.Binding NExprLoc] -> Maybe NExprLoc
findBinding name = foldr check Nothing
  where
    check (Nix.NamedVar (StaticKey k :| []) expr _) acc
      | varNameText k == name = Just expr
      | otherwise = acc
    check _ acc = acc

-- | Extract a string literal
extractStringLit :: NExprLoc -> Maybe Text
extractStringLit expr = case unwrapExpr expr of
  NStr (DoubleQuoted [Plain t]) -> Just t
  NStr (Indented _ [Plain t]) -> Just t
  _ -> Nothing

-- | Extract a bool literal
extractBoolLit :: NExprLoc -> Maybe Bool
extractBoolLit expr = case unwrapExpr expr of
  NConstant (NBool b) -> Just b
  _ -> Nothing

-- | Extract text from VarName
varNameText :: VarName -> Text
varNameText = coerce
