{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

-- |
-- Module      : Render.Schema.Build
-- Description : Build schema from facts and substitution
--
-- Takes the raw facts and solved type substitution and produces
-- the final schema with resolved types.
module Render.Schema.Build
  ( buildSchema,
    resolveType,
  )
where

import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import Render.Types

-- | Build schema from facts and type substitution
buildSchema :: [Fact] -> Subst -> Schema
buildSchema facts subst =
  Schema
    { schemaEnv = buildEnvSchema facts subst,
      schemaConfig = buildConfigSchema facts subst,
      schemaCommands = buildCommandSchema facts,
      schemaStorePaths = collectStorePaths facts,
      schemaBareCommands = collectBareCommands facts,
      schemaDynamicCommands = collectDynamicCommands facts
    }

-- | Build environment variable schema
buildEnvSchema :: [Fact] -> Subst -> Map Text EnvSpec
buildEnvSchema facts subst = Map.fromListWith mergeEnvSpec (concatMap go facts)
  where
    go = \case
      DefaultIs var lit span ->
        [(var, EnvSpec (resolveType subst var) False (Just lit) span)]
      DefaultFrom var _ span ->
        [(var, EnvSpec (resolveType subst var) False Nothing span)]
      Required var span ->
        [(var, EnvSpec (resolveType subst var) True Nothing span)]
      AssignLit var lit span ->
        [(var, EnvSpec (resolveType subst var) False (Just lit) span)]
      AssignFrom var _ span ->
        [(var, EnvSpec (resolveType subst var) False Nothing span)]
      ConfigAssign _ var _ span ->
        [(var, EnvSpec (resolveType subst var) False Nothing span)]
      -- Command argument usage: infer type from builtin database
      CmdArg _ _ var span ->
        [(var, EnvSpec (resolveType subst var) False Nothing span)]
      _ -> []

-- | Merge two env specs (prefer required, keep first default)
mergeEnvSpec :: EnvSpec -> EnvSpec -> EnvSpec
mergeEnvSpec e1 e2 =
  EnvSpec
    { envType = envType e1, -- take first (arbitrary)
      envRequired = envRequired e1 || envRequired e2,
      envDefault = envDefault e1 <|> envDefault e2,
      envSpan = envSpan e1
    }
  where
    Nothing <|> b = b
    a <|> _ = a

-- | Build config schema
buildConfigSchema :: [Fact] -> Subst -> Map ConfigPath ConfigSpec
buildConfigSchema facts subst = Map.fromList (concatMap go facts)
  where
    go = \case
      ConfigAssign path var _ span ->
        [(path, ConfigSpec (resolveType subst var) (Just var) span)]
      ConfigLit path lit span ->
        [(path, ConfigSpec (literalType lit) Nothing span)]
      _ -> []

-- | Build command schema
buildCommandSchema :: [Fact] -> [CommandSpec]
buildCommandSchema facts = concatMap go facts
  where
    go = \case
      UsesStorePath sp span ->
        [CommandSpec (extractName sp) (Just sp) span]
      BareCommand cmd span ->
        [CommandSpec cmd Nothing span]
      _ -> []
    extractName :: StorePath -> Text
    extractName (StorePath p) =
      -- /nix/store/hash-name/bin/cmd -> cmd
      case reverse (T.splitOn "/" p) of
        (cmd : _) | not (T.null cmd) -> cmd
        _ -> p

-- | Collect store paths
collectStorePaths :: [Fact] -> Set StorePath
collectStorePaths facts = Set.fromList [sp | UsesStorePath sp _ <- facts]

-- | Collect bare commands
collectBareCommands :: [Fact] -> [Text]
collectBareCommands facts = [cmd | BareCommand cmd _ <- facts]

-- | Collect dynamic commands
collectDynamicCommands :: [Fact] -> [Text]
collectDynamicCommands facts = [var | DynamicCommand var _ <- facts]

-- | Resolve a variable's type from substitution
resolveType :: Subst -> Text -> Type
resolveType subst var =
  applyDefaults (applySubst subst (TVar (TypeVar var)))

-- | Apply defaults: TNumeric -> TInt, TVar -> TString
applyDefaults :: Type -> Type
applyDefaults = \case
  TNumeric -> TInt
  TVar _ -> TString -- unresolved becomes string
  t -> t
