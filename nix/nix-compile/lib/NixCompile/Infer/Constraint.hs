{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- |
-- Module      : NixCompile.Infer.Constraint
-- Description : Generate type constraints from facts
--
-- Transforms observations (Facts) into equality constraints for unification.
module NixCompile.Infer.Constraint
  ( factsToConstraints,
    factToConstraints,
  )
where

import Data.Text (Text)
import NixCompile.Bash.Builtins (lookupArgType)
import NixCompile.Types

-- | Convert all facts to constraints
factsToConstraints :: [Fact] -> [Constraint]
factsToConstraints = concatMap factToConstraints

-- | Convert a single fact to constraints
factToConstraints :: Fact -> [Constraint]
factToConstraints = \case
  -- VAR has a literal default: VAR ~ type(literal)
  DefaultIs var lit _ ->
    [TVar (TypeVar var) :~: literalType lit]
  -- VAR defaults to OTHER: VAR ~ OTHER
  DefaultFrom var other _ ->
    [TVar (TypeVar var) :~: TVar (TypeVar other)]
  -- VAR is required: no type constraint, just existence
  Required _ _ ->
    []
  -- VAR = OTHER: VAR ~ OTHER
  AssignFrom var other _ ->
    [TVar (TypeVar var) :~: TVar (TypeVar other)]
  -- VAR = literal: VAR ~ type(literal)
  AssignLit var lit _ ->
    [TVar (TypeVar var) :~: literalType lit]
  -- config.x.y = $VAR or "$VAR": no type constraint on VAR
  -- The config inherits the variable's type; quoting is about bash word-splitting,
  -- not about type. Type flows from definition (DefaultIs/AssignLit) to usage.
  ConfigAssign _ _ _ _ ->
    []
  -- config.x.y = literal: no variable constraint
  ConfigLit _ _ _ ->
    []
  -- Command arg with known type: look up in builtins
  CmdArg cmd argName varName _ ->
    case lookupArgType cmd argName of
      Just ty -> [TVar (TypeVar varName) :~: ty]
      Nothing -> []
  -- Store path usage: no type constraint
  UsesStorePath _ _ ->
    []
  -- Bare command: no type constraint
  BareCommand _ _ ->
    []
  -- Dynamic command: no type constraint
  DynamicCommand _ _ ->
    []
