{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

-- |
-- Module      : Render.Infer.Unify
-- Description : Hindley-Milner unification for bash types
--
-- Standard unification algorithm. Nothing fancy.
--
-- The key insight: we don't need polymorphism. Bash variables are monomorphic.
-- So this is just first-order unification, which is simple and decidable.
module Render.Infer.Unify
  ( unify,
    unifyAll,
    solve,
  )
where

import Control.Monad (foldM)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Render.Types

-- | Unify two types, producing a substitution
unify :: Type -> Type -> Either TypeError Subst
unify t1 t2 = case (t1, t2) of
  -- Same concrete types
  (TInt, TInt) -> Right emptySubst
  (TString, TString) -> Right emptySubst
  (TBool, TBool) -> Right emptySubst
  (TPath, TPath) -> Right emptySubst
  -- TNumeric unifies with Int or Bool
  (TNumeric, TInt) -> Right emptySubst
  (TInt, TNumeric) -> Right emptySubst
  (TNumeric, TBool) -> Right emptySubst
  (TBool, TNumeric) -> Right emptySubst
  (TNumeric, TNumeric) -> Right emptySubst
  -- Type variable on left: bind it
  (TVar v, t) -> bindVar v t
  -- Type variable on right: bind it
  (t, TVar v) -> bindVar v t
  -- Mismatch
  _ -> Left (Mismatch t1 t2 emptySpan)
  where
    emptySpan = Span (Loc 0 0) (Loc 0 0) Nothing

-- | Bind a type variable, checking for occurs
bindVar :: TypeVar -> Type -> Either TypeError Subst
bindVar v t
  | t == TVar v = Right emptySubst -- v ~ v is trivial
  | occursIn v t = Left (OccursCheck v t emptySpan)
  | otherwise = Right (singleSubst v t)
  where
    emptySpan = Span (Loc 0 0) (Loc 0 0) Nothing

-- | Does a type variable occur in a type?
-- For our simple type language, this only matters for TVar
occursIn :: TypeVar -> Type -> Bool
occursIn v = \case
  TVar v' -> v == v'
  _ -> False

-- | Unify a list of constraints, accumulating substitutions
unifyAll :: [Constraint] -> Either TypeError Subst
unifyAll = foldM go emptySubst
  where
    go s (t1 :~: t2) = do
      let t1' = applySubst s t1
          t2' = applySubst s t2
      s' <- unify t1' t2'
      Right (composeSubst s' s)

-- | Solve constraints and return final substitution
-- This is the main entry point
solve :: [Constraint] -> Either TypeError Subst
solve = unifyAll
