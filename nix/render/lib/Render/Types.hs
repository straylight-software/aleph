{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- |
-- Module      : Render.Types
-- Description : Core types for render.nix
--
-- The type system for bash environment inference.
--
-- Design:
--   - Types are simple: Int, String, Bool, Path, or unknown (TVar)
--   - Constraints are equality: T1 :~: T2
--   - Facts are observations from parsing: "VAR has default 8080"
--   - Schema is the final output: env vars, config structure, commands
module Render.Types
  ( -- * Types
    Type (..),
    TypeVar (..),

    -- * Constraints
    Constraint (..),
    Subst,
    emptySubst,
    singleSubst,
    composeSubst,
    applySubst,

    -- * Source locations
    Loc (..),
    Span (..),

    -- * Literals
    Literal (..),
    literalType,

    -- * Facts (observations from parsing)
    Fact (..),
    Quoted (..),

    -- * Config paths
    ConfigPath,

    -- * Commands
    Command (..),
    Arg (..),

    -- * Store paths
    StorePath (..),
    isStorePath,

    -- * Schema (final output)
    Schema (..),
    EnvSpec (..),
    ConfigSpec (..),
    CommandSpec (..),
    emptySchema,
    mergeSchemas,

    -- * Scripts
    Script (..),

    -- * Errors
    TypeError (..),
    LintError (..),
    Severity (..),
  )
where

import Data.Aeson (FromJSON, ToJSON)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import GHC.Generics (Generic)

-- ============================================================================
-- Types
-- ============================================================================

-- | Type variables for unification
newtype TypeVar = TypeVar {unTypeVar :: Text}
  deriving stock (Eq, Ord, Show, Generic)
  deriving newtype (FromJSON, ToJSON)

-- | The type language
data Type
  = TInt
  | TString
  | TBool
  | TPath -- Nix store path
  | TNumeric -- Int or Bool (unquoted in config.*)
  | TVar TypeVar -- Unification variable
  deriving stock (Eq, Ord, Show, Generic)

instance FromJSON Type

instance ToJSON Type

-- ============================================================================
-- Constraints
-- ============================================================================

-- | Equality constraint: these two types must unify
data Constraint = Type :~: Type
  deriving stock (Eq, Show, Generic)

infix 4 :~:

-- | Substitution: mapping from type variables to types
type Subst = Map TypeVar Type

emptySubst :: Subst
emptySubst = Map.empty

singleSubst :: TypeVar -> Type -> Subst
singleSubst = Map.singleton

-- | Compose substitutions: apply s1 then s2
composeSubst :: Subst -> Subst -> Subst
composeSubst s1 s2 = Map.map (applySubst s1) s2 `Map.union` s1

-- | Apply substitution to a type
applySubst :: Subst -> Type -> Type
applySubst s = \case
  TVar v -> Map.findWithDefault (TVar v) v s
  t -> t

-- ============================================================================
-- Source Locations
-- ============================================================================

-- | Source location
data Loc = Loc
  { locLine :: !Int,
    locCol :: !Int
  }
  deriving stock (Eq, Ord, Show, Generic)

instance FromJSON Loc

instance ToJSON Loc

-- | Source span
data Span = Span
  { spanStart :: !Loc,
    spanEnd :: !Loc,
    spanFile :: !(Maybe FilePath)
  }
  deriving stock (Eq, Show, Generic)

instance FromJSON Span

instance ToJSON Span

-- ============================================================================
-- Literals
-- ============================================================================

-- | Literal values observed in scripts
data Literal
  = LitInt !Int
  | LitString !Text
  | LitBool !Bool
  | LitPath !StorePath
  deriving stock (Eq, Show, Generic)

instance FromJSON Literal

instance ToJSON Literal

-- | Get the type of a literal
literalType :: Literal -> Type
literalType = \case
  LitInt _ -> TInt
  LitString _ -> TString
  LitBool _ -> TBool
  LitPath _ -> TPath

-- ============================================================================
-- Facts
-- ============================================================================

-- | Whether a value was quoted in bash
data Quoted = Quoted | Unquoted
  deriving stock (Eq, Show, Generic)

instance FromJSON Quoted

instance ToJSON Quoted

-- | A path in config.* namespace
-- e.g., ["server", "port"] for config.server.port
type ConfigPath = [Text]

-- | Facts extracted from parsing
data Fact
  = -- | VAR="${VAR:-default}" - variable has a default
    DefaultIs !Text !Literal !Span
  | -- | VAR="${VAR:-$OTHER}" - variable defaults to another var
    DefaultFrom !Text !Text !Span
  | -- | VAR="${VAR:?}" - variable is required
    Required !Text !Span
  | -- | VAR="$OTHER" - simple assignment from another var
    AssignFrom !Text !Text !Span
  | -- | VAR="literal" - assignment from literal
    AssignLit !Text !Literal !Span
  | -- | config.x.y=$VAR - config assignment
    ConfigAssign !ConfigPath !Text !Quoted !Span
  | -- | config.x.y=literal - config literal
    ConfigLit !ConfigPath !Literal !Span
  | -- | Command invocation with known arg
    CmdArg !Text !Text !Text !Span -- cmd, argname, varname, span
  | -- | Store path usage
    UsesStorePath !StorePath !Span
  | -- | Bare command (not a store path)
    BareCommand !Text !Span
  | -- | Dynamic command ($VAR as command)
    DynamicCommand !Text !Span
  deriving stock (Eq, Show, Generic)

instance FromJSON Fact

instance ToJSON Fact

-- ============================================================================
-- Commands
-- ============================================================================

-- | A command argument
data Arg
  = ArgLit !Text
  | ArgVar !Text
  | ArgFlag !Text
  deriving stock (Eq, Show, Generic)

instance FromJSON Arg

instance ToJSON Arg

-- | A command invocation
data Command = Command
  { cmdName :: !Text,
    cmdPath :: !(Maybe StorePath),
    cmdArgs :: ![Arg],
    cmdSpan :: !Span
  }
  deriving stock (Eq, Show, Generic)

instance FromJSON Command

instance ToJSON Command

-- ============================================================================
-- Store Paths
-- ============================================================================

-- | A Nix store path
newtype StorePath = StorePath {unStorePath :: Text}
  deriving stock (Eq, Ord, Show, Generic)
  deriving newtype (FromJSON, ToJSON)

-- | Check if a text looks like a store path
isStorePath :: Text -> Bool
isStorePath t = "/nix/store/" `T.isPrefixOf` t

-- ============================================================================
-- Schema
-- ============================================================================

-- | Environment variable specification
data EnvSpec = EnvSpec
  { envType :: !Type,
    envRequired :: !Bool,
    envDefault :: !(Maybe Literal),
    envSpan :: !Span
  }
  deriving stock (Eq, Show, Generic)

instance FromJSON EnvSpec

instance ToJSON EnvSpec

-- | Config field specification
data ConfigSpec = ConfigSpec
  { cfgType :: !Type,
    cfgFrom :: !(Maybe Text), -- source env var, if any
    cfgSpan :: !Span
  }
  deriving stock (Eq, Show, Generic)

instance FromJSON ConfigSpec

instance ToJSON ConfigSpec

-- | Command specification
data CommandSpec = CommandSpec
  { cmdSpecName :: !Text,
    cmdSpecPath :: !(Maybe StorePath),
    cmdSpecSpan :: !Span
  }
  deriving stock (Eq, Show, Generic)

instance FromJSON CommandSpec

instance ToJSON CommandSpec

-- | Complete schema for a script
data Schema = Schema
  { schemaEnv :: !(Map Text EnvSpec),
    schemaConfig :: !(Map ConfigPath ConfigSpec),
    schemaCommands :: ![CommandSpec],
    schemaStorePaths :: !(Set StorePath),
    schemaBareCommands :: ![Text],
    schemaDynamicCommands :: ![Text]
  }
  deriving stock (Eq, Show, Generic)

instance FromJSON Schema

instance ToJSON Schema

emptySchema :: Schema
emptySchema =
  Schema
    { schemaEnv = Map.empty,
      schemaConfig = Map.empty,
      schemaCommands = [],
      schemaStorePaths = Set.empty,
      schemaBareCommands = [],
      schemaDynamicCommands = []
    }

-- | Merge two schemas (for composition)
mergeSchemas :: Schema -> Schema -> Schema
mergeSchemas s1 s2 =
  Schema
    { schemaEnv = schemaEnv s1 `Map.union` schemaEnv s2,
      schemaConfig = schemaConfig s1 `Map.union` schemaConfig s2,
      schemaCommands = schemaCommands s1 ++ schemaCommands s2,
      schemaStorePaths = schemaStorePaths s1 `Set.union` schemaStorePaths s2,
      schemaBareCommands = schemaBareCommands s1 ++ schemaBareCommands s2,
      schemaDynamicCommands = schemaDynamicCommands s1 ++ schemaDynamicCommands s2
    }

-- ============================================================================
-- Scripts
-- ============================================================================

-- | A parsed script with its schema
data Script = Script
  { scriptSource :: !Text,
    scriptFacts :: ![Fact],
    scriptSchema :: !Schema
  }
  deriving stock (Eq, Show, Generic)

instance FromJSON Script

instance ToJSON Script

-- ============================================================================
-- Errors
-- ============================================================================

-- | Type unification error
data TypeError
  = Mismatch !Type !Type !Span
  | OccursCheck !TypeVar !Type !Span
  | Ambiguous !TypeVar !Span
  deriving stock (Eq, Show, Generic)

instance FromJSON TypeError

instance ToJSON TypeError

-- | Severity levels
data Severity
  = SevError
  | SevWarning
  | SevInfo
  deriving stock (Eq, Ord, Show, Generic)

instance FromJSON Severity

instance ToJSON Severity

-- | Lint error
data LintError = LintError
  { lintCode :: !Text, -- e.g., "ALEPH-E001"
    lintMessage :: !Text,
    lintSeverity :: !Severity,
    lintSpan :: !Span,
    lintSuggestion :: !(Maybe Text)
  }
  deriving stock (Eq, Show, Generic)

instance FromJSON LintError

instance ToJSON LintError
