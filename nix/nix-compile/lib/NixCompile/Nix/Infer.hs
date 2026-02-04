{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- |
-- Module      : NixCompile.Nix.Infer
-- Description : Type inference for Nix expressions
--
-- Hindley-Milner style type inference for a subset of Nix.
-- 
-- We infer types from:
--   - Literals: 42 : Int, "hello" : String, true : Bool
--   - Defaults: { port ? 8080 } → port : Int
--   - Operators: a + b with a : Int → b : Int, result : Int
--   - Builtins: toString : (Int | Float | Bool | Path) -> String
--   - Function application: f x where f : A -> B, x : A → result : B
--
-- The inference produces:
--   - Type annotations for function parameters
--   - Type annotations for let bindings
--   - Type signatures for top-level definitions
module NixCompile.Nix.Infer
  ( -- * Inference
    inferExpr,
    inferFile,
    
    -- * Environment
    TypeEnv (..),
    emptyEnv,
    builtinEnv,
    
    -- * Results
    InferResult (..),
    Binding (..),
  )
where

import Control.Monad (foldM, forM, replicateM)
import Control.Monad.State.Strict
import Data.Coerce (coerce)
import Data.Fix (Fix (..))
import Data.Functor.Compose (Compose (..))
import Data.List.NonEmpty (NonEmpty (..))
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe, mapMaybe)
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import Nix.Atoms (NAtom (..))
import Nix.Expr.Types hiding (Binding)
import qualified Nix.Expr.Types as Nix
import Nix.Expr.Types.Annotated
import Nix.Parser (parseNixFileLoc, parseNixTextLoc)
import qualified Nix.Utils as Nix
import NixCompile.Nix.Types
import NixCompile.Types (Loc (..), Span (..))

-- ============================================================================
-- Environment
-- ============================================================================

-- | Type environment: maps variable names to their type schemes
newtype TypeEnv = TypeEnv { unTypeEnv :: Map Text Scheme }
  deriving (Eq, Show)

emptyEnv :: TypeEnv
emptyEnv = TypeEnv Map.empty

extendEnv :: Text -> Scheme -> TypeEnv -> TypeEnv
extendEnv name scheme (TypeEnv m) = TypeEnv (Map.insert name scheme m)

lookupEnv :: Text -> TypeEnv -> Maybe Scheme
lookupEnv name (TypeEnv m) = Map.lookup name m

-- | Built-in function types
builtinEnv :: TypeEnv
builtinEnv = TypeEnv $ Map.fromList
  [ -- String functions
    ("toString", mono $ TFun (TUnion [TInt, TFloat, TBool, TPath, TString]) TString)
  , ("baseNameOf", mono $ TFun TPath TString)
  , ("dirOf", mono $ TFun TPath TPath)
  , ("stringLength", mono $ TFun TString TInt)
  , ("substring", mono $ TFun TInt (TFun TInt (TFun TString TString)))
  , ("replaceStrings", mono $ TFun (TList TString) (TFun (TList TString) (TFun TString TString)))
  
  -- List functions  
  , ("head", Forall [TypeVar 0] $ TFun (TList (TVar (TypeVar 0))) (TVar (TypeVar 0)))
  , ("tail", Forall [TypeVar 0] $ TFun (TList (TVar (TypeVar 0))) (TList (TVar (TypeVar 0))))
  , ("length", Forall [TypeVar 0] $ TFun (TList (TVar (TypeVar 0))) TInt)
  , ("elemAt", Forall [TypeVar 0] $ TFun (TList (TVar (TypeVar 0))) (TFun TInt (TVar (TypeVar 0))))
  , ("filter", Forall [TypeVar 0] $ TFun (TFun (TVar (TypeVar 0)) TBool) (TFun (TList (TVar (TypeVar 0))) (TList (TVar (TypeVar 0)))))
  , ("map", Forall [TypeVar 0, TypeVar 1] $ TFun (TFun (TVar (TypeVar 0)) (TVar (TypeVar 1))) (TFun (TList (TVar (TypeVar 0))) (TList (TVar (TypeVar 1)))))
  , ("foldl'", Forall [TypeVar 0, TypeVar 1] $ TFun (TFun (TVar (TypeVar 0)) (TFun (TVar (TypeVar 1)) (TVar (TypeVar 0)))) (TFun (TVar (TypeVar 0)) (TFun (TList (TVar (TypeVar 1))) (TVar (TypeVar 0)))))
  , ("concatLists", Forall [TypeVar 0] $ TFun (TList (TList (TVar (TypeVar 0)))) (TList (TVar (TypeVar 0))))
  , ("concatMap", Forall [TypeVar 0, TypeVar 1] $ TFun (TFun (TVar (TypeVar 0)) (TList (TVar (TypeVar 1)))) (TFun (TList (TVar (TypeVar 0))) (TList (TVar (TypeVar 1)))))
  
  -- Attrset functions
  , ("attrNames", mono $ TFun (TAttrsOpen Map.empty) (TList TString))
  , ("attrValues", Forall [TypeVar 0] $ TFun (TAttrsOpen (Map.singleton "_" (TVar (TypeVar 0)))) (TList (TVar (TypeVar 0))))
  , ("hasAttr", mono $ TFun TString (TFun (TAttrsOpen Map.empty) TBool))
  , ("getAttr", Forall [TypeVar 0] $ TFun TString (TFun (TAttrsOpen Map.empty) (TVar (TypeVar 0))))
  , ("removeAttrs", mono $ TFun (TAttrsOpen Map.empty) (TFun (TList TString) (TAttrsOpen Map.empty)))
  , ("listToAttrs", Forall [TypeVar 0] $ TFun (TList (TAttrs (Map.fromList [("name", TString), ("value", TVar (TypeVar 0))]))) (TAttrsOpen Map.empty))
  
  -- Type checking
  , ("isNull", mono $ TFun TAny TBool)
  , ("isInt", mono $ TFun TAny TBool)
  , ("isFloat", mono $ TFun TAny TBool)
  , ("isBool", mono $ TFun TAny TBool)
  , ("isString", mono $ TFun TAny TBool)
  , ("isList", mono $ TFun TAny TBool)
  , ("isAttrs", mono $ TFun TAny TBool)
  , ("isFunction", mono $ TFun TAny TBool)
  , ("isPath", mono $ TFun TAny TBool)
  
  -- Arithmetic
  , ("add", mono $ TFun TInt (TFun TInt TInt))
  , ("sub", mono $ TFun TInt (TFun TInt TInt))
  , ("mul", mono $ TFun TInt (TFun TInt TInt))
  , ("div", mono $ TFun TInt (TFun TInt TInt))
  
  -- Comparison
  , ("lessThan", mono $ TFun TInt (TFun TInt TBool))
  
  -- Import
  , ("import", mono $ TFun TPath TAny)
  
  -- Derivation
  , ("derivation", mono $ TFun (TAttrsOpen Map.empty) TDerivation)
  
  -- Misc
  , ("throw", Forall [TypeVar 0] $ TFun TString (TVar (TypeVar 0)))
  , ("abort", Forall [TypeVar 0] $ TFun TString (TVar (TypeVar 0)))
  , ("trace", Forall [TypeVar 0] $ TFun TString (TFun (TVar (TypeVar 0)) (TVar (TypeVar 0))))
  , ("seq", Forall [TypeVar 0, TypeVar 1] $ TFun (TVar (TypeVar 0)) (TFun (TVar (TypeVar 1)) (TVar (TypeVar 1))))
  , ("deepSeq", Forall [TypeVar 0, TypeVar 1] $ TFun (TVar (TypeVar 0)) (TFun (TVar (TypeVar 1)) (TVar (TypeVar 1))))
  , ("tryEval", Forall [TypeVar 0] $ TFun (TVar (TypeVar 0)) (TAttrs (Map.fromList [("success", TBool), ("value", TVar (TypeVar 0))])))
  ]
  where
    mono t = Forall [] t

-- ============================================================================
-- Inference State
-- ============================================================================

-- | Inference state
data InferState = InferState
  { inferSupply :: !Int  -- Fresh type variable supply
  , inferSubst :: !Subst -- Current substitution
  }

type Infer a = State InferState a

runInfer :: Infer a -> a
runInfer m = evalState m (InferState 0 emptySubst)

-- | Generate a fresh type variable
freshVar :: Infer NixType
freshVar = do
  s <- get
  put s { inferSupply = inferSupply s + 1 }
  pure $ TVar (TypeVar (inferSupply s))

-- | Apply current substitution to a type
applyCurrentSubst :: NixType -> Infer NixType
applyCurrentSubst t = do
  s <- gets inferSubst
  pure $ applySubst s t

-- | Add a substitution
addSubst :: TypeVar -> NixType -> Infer ()
addSubst v t = modify $ \s ->
  s { inferSubst = composeSubst (singleSubst v t) (inferSubst s) }

-- ============================================================================
-- Unification
-- ============================================================================

-- | Unify two types
unify :: NixType -> NixType -> Infer ()
unify t1 t2 = do
  t1' <- applyCurrentSubst t1
  t2' <- applyCurrentSubst t2
  unify' t1' t2'

unify' :: NixType -> NixType -> Infer ()
unify' t1 t2 = case (t1, t2) of
  (TVar v, t) -> bindVar v t
  (t, TVar v) -> bindVar v t
  (TAny, _) -> pure ()
  (_, TAny) -> pure ()
  (TInt, TInt) -> pure ()
  (TFloat, TFloat) -> pure ()
  (TBool, TBool) -> pure ()
  (TString, TString) -> pure ()
  (TPath, TPath) -> pure ()
  (TNull, TNull) -> pure ()
  (TDerivation, TDerivation) -> pure ()
  (TList a, TList b) -> unify a b
  (TFun a1 b1, TFun a2 b2) -> unify a1 a2 >> unify b1 b2
  (TAttrs m1, TAttrs m2) -> unifyAttrs m1 m2
  (TAttrsOpen m1, TAttrsOpen m2) -> unifyAttrsOpenOpen m1 m2
  (TAttrs m1, TAttrsOpen m2) -> unifyAttrsClosedOpen m1 m2
  (TAttrsOpen m1, TAttrs m2) -> unifyAttrsClosedOpen m2 m1
  (TUnion ts, t) -> unifyUnion ts t
  (t, TUnion ts) -> unifyUnion ts t
  _ -> pure () -- Mismatch, but don't fail (we're lenient)

bindVar :: TypeVar -> NixType -> Infer ()
bindVar v t
  | t == TVar v = pure ()
  | occursCheck v t = pure () -- Occurs check, skip
  | otherwise = addSubst v t

occursCheck :: TypeVar -> NixType -> Bool
occursCheck v = \case
  TVar v' -> v == v'
  TList t -> occursCheck v t
  TFun a b -> occursCheck v a || occursCheck v b
  TAttrs m -> any (occursCheck v) (Map.elems m)
  TAttrsOpen m -> any (occursCheck v) (Map.elems m)
  TUnion ts -> any (occursCheck v) ts
  _ -> False

unifyAttrs :: Map Text NixType -> Map Text NixType -> Infer ()
unifyAttrs m1 m2 = do
  -- Closed rows must match exactly
  if Map.keysSet m1 /= Map.keysSet m2
    then error "Type error: Attribute set mismatch (closed rows)" -- TODO: proper error
    else mapM_ (uncurry unify) (Map.elems (Map.intersectionWith (,) m1 m2))

unifyAttrsOpenOpen :: Map Text NixType -> Map Text NixType -> Infer ()
unifyAttrsOpenOpen m1 m2 = do
  -- Open vs Open: unify common fields
  let common = Map.intersectionWith (,) m1 m2
  mapM_ (uncurry unify) (Map.elems common)

unifyAttrsClosedOpen :: Map Text NixType -> Map Text NixType -> Infer ()
unifyAttrsClosedOpen closed open = do
  -- Closed vs Open: Open must be subset of Closed
  if not (Map.keysSet open `Set.isSubsetOf` Map.keysSet closed)
    then error "Type error: Closed attrset missing required fields from open attrset"
    else do
      -- Unify the common fields (which is all of open)
      let common = Map.intersectionWith (,) closed open
      mapM_ (uncurry unify) (Map.elems common)

unifyUnion :: [NixType] -> NixType -> Infer ()
unifyUnion ts t = case ts of
  [] -> pure ()
  [t'] -> unify t' t
  _ -> pure () -- Can't easily unify with unions, be lenient

-- ============================================================================
-- Instantiation
-- ============================================================================

-- | Instantiate a type scheme with fresh variables
instantiate :: Scheme -> Infer NixType
instantiate (Forall vars t) = do
  freshVars <- mapM (const freshVar) vars
  let subst = Map.fromList (zip vars freshVars)
  pure $ applySubst subst t

-- ============================================================================
-- Inference
-- ============================================================================

-- | Infer the type of an expression
infer :: TypeEnv -> NExprLoc -> Infer NixType
infer env (Fix (Compose (AnnUnit _ expr))) = case expr of
  -- Literals
  NConstant atom -> pure $ atomType atom
  
  -- Strings (could contain interpolations)
  NStr _ -> pure TString
  
  -- Paths
  NLiteralPath _ -> pure TPath
  NEnvPath _ -> pure TPath
  
  -- Variables
  NSym name -> case lookupEnv (varNameText name) env of
    Just scheme -> instantiate scheme
    Nothing -> freshVar -- Unknown variable, assign fresh
  
  -- Lists
  NList [] -> do
    elemType <- freshVar
    pure $ TList elemType
  NList (x:xs) -> do
    elemType <- infer env x
    mapM_ (\e -> infer env e >>= unify elemType) xs
    TList <$> applyCurrentSubst elemType
  
  -- Attribute sets
  NSet recursive bindings -> do
    fields <- inferBindings (recursive == Recursive) env bindings
    pure $ TAttrs (Map.fromList fields)
  
  -- Let bindings
  NLet bindings body -> inferLet env bindings body
  
  -- If expression
  NIf cond thenE elseE -> do
    condT <- infer env cond
    unify condT TBool
    thenT <- infer env thenE
    elseT <- infer env elseE
    unify thenT elseT
    applyCurrentSubst thenT
  
  -- With expression
  NWith scope body -> do
    _ <- infer env scope -- Just check scope, don't use its type
    infer env body
  
  -- Assert
  NAssert cond body -> do
    condT <- infer env cond
    unify condT TBool
    infer env body
  
  -- Lambda
  NAbs params body -> inferLambda env params body
  
  -- Application
  NApp func arg -> do
    funcT <- infer env func
    argT <- infer env arg
    resultT <- freshVar
    unify funcT (TFun argT resultT)
    applyCurrentSubst resultT
  
  -- Selection (a.b)
  NSelect _ base (attr :| _) -> do
    baseT <- infer env base
    resultT <- freshVar
    -- For now, just return fresh - proper would track attrset structure
    pure resultT
  
  -- Has attribute
  NHasAttr base _ -> do
    _ <- infer env base
    pure TBool
  
  -- Unary operators
  NUnary op e -> do
    t <- infer env e
    case op of
      NNeg -> unify t TInt >> pure TInt
      NNot -> unify t TBool >> pure TBool
  
  -- Binary operators
  NBinary op left right -> inferBinary env op left right
  
  -- Holes (shouldn't appear normally)
  NSynHole _ -> freshVar

-- | Infer type of a binary operation
inferBinary :: TypeEnv -> NBinaryOp -> NExprLoc -> NExprLoc -> Infer NixType
inferBinary env op left right = do
  leftT <- infer env left
  rightT <- infer env right
  case op of
    -- Equality: any types, but should be same
    NEq -> unify leftT rightT >> pure TBool
    NNEq -> unify leftT rightT >> pure TBool
    
    -- Comparison: numeric
    NLt -> unify leftT TInt >> unify rightT TInt >> pure TBool
    NLte -> unify leftT TInt >> unify rightT TInt >> pure TBool
    NGt -> unify leftT TInt >> unify rightT TInt >> pure TBool
    NGte -> unify leftT TInt >> unify rightT TInt >> pure TBool
    
    -- Logical
    NAnd -> unify leftT TBool >> unify rightT TBool >> pure TBool
    NOr -> unify leftT TBool >> unify rightT TBool >> pure TBool
    NImpl -> unify leftT TBool >> unify rightT TBool >> pure TBool
    
    -- Arithmetic
    NPlus -> do
      -- Could be int+int, string+string, path+string, list+list
      resultT <- freshVar
      pure resultT -- Be lenient
    NMinus -> unify leftT TInt >> unify rightT TInt >> pure TInt
    NMult -> unify leftT TInt >> unify rightT TInt >> pure TInt
    NDiv -> unify leftT TInt >> unify rightT TInt >> pure TInt
    
    -- List concatenation
    NConcat -> do
      unify leftT rightT
      applyCurrentSubst leftT
    
    -- Attrset update
    NUpdate -> do
      unify leftT rightT
      applyCurrentSubst leftT

-- | Infer type of a lambda
inferLambda :: TypeEnv -> Params NExprLoc -> NExprLoc -> Infer NixType
inferLambda env params body = case params of
  -- Simple parameter: x: body
  Param name -> do
    paramT <- freshVar
    let env' = extendEnv (varNameText name) (Forall [] paramT) env
    resultT <- infer env' body
    paramT' <- applyCurrentSubst paramT
    pure $ TFun paramT' resultT
  
  -- Pattern: { a, b ? default, ... }: body
  ParamSet mName variadic paramList -> do
    -- Infer types from defaults
    paramTypes <- forM paramList $ \(name, mDefault) -> do
      t <- case mDefault of
        Just defaultExpr -> infer env defaultExpr
        Nothing -> freshVar
      pure (varNameText name, t)
    
    let attrsT = if variadic == Variadic
                   then TAttrsOpen (Map.fromList paramTypes)
                   else TAttrs (Map.fromList paramTypes)
                   
    let env' = foldr (\(n, t) e -> extendEnv n (Forall [] t) e) env paramTypes
    
    -- Also bind the @ pattern if present
    let env'' = case mName of
          Just name -> extendEnv (varNameText name) (Forall [] attrsT) env'
          Nothing -> env'
    
    resultT <- infer env'' body
    pure $ TFun attrsT resultT

-- | Infer types for bindings in an attrset
inferBindings :: Bool -> TypeEnv -> [Nix.Binding NExprLoc] -> Infer [(Text, NixType)]
inferBindings recursive env bindings
  | recursive = do
      -- Recursive: bind all names to fresh vars first
      let names = mapMaybe bindingName bindings
      freshVars <- replicateM (length names) freshVar
      let env' = foldr (\(n, t) e -> extendEnv n (Forall [] t) e) env (zip names freshVars)
      
      -- Infer bodies and unify
      forM (zip bindings freshVars) $ \(binding, typeVar) -> do
        case binding of
          Nix.NamedVar (StaticKey name :| []) expr _ -> do
            t <- infer env' expr
            unify typeVar t
            pure (varNameText name, t)
          _ -> pure ("", typeVar) -- Skip complex bindings for now
  | otherwise = foldM go [] bindings
  where
    go acc binding = case binding of
      Nix.NamedVar (StaticKey name :| []) expr _ -> do
        t <- infer env expr
        pure $ acc ++ [(varNameText name, t)]
      _ -> pure acc

    bindingName (Nix.NamedVar (StaticKey name :| []) _ _) = Just (varNameText name)
    bindingName _ = Nothing

-- | Infer type for a single binding (used in non-recursive Let)
inferBinding :: TypeEnv -> Nix.Binding NExprLoc -> Infer TypeEnv
inferBinding env binding = case binding of
  Nix.NamedVar (StaticKey name :| []) expr _ -> do
    t <- infer env expr
    pure $ extendEnv (varNameText name) (Forall [] t) env
  _ -> pure env

-- | Infer types for a Let block (recursive)
inferLet :: TypeEnv -> [Nix.Binding NExprLoc] -> NExprLoc -> Infer NixType
inferLet env bindings body = do
  -- Recursive let is like recursive attrset
  let names = mapMaybe bindingName bindings
  freshVars <- replicateM (length names) freshVar
  let env' = foldr (\(n, t) e -> extendEnv n (Forall [] t) e) env (zip names freshVars)
  
  -- Infer bodies and unify
  mapM_ (\(binding, typeVar) -> do
    case binding of
      Nix.NamedVar (StaticKey _ :| []) expr _ -> do
        t <- infer env' expr
        unify typeVar t
      _ -> pure ()
    ) (zip bindings freshVars)
    
  infer env' body
  where
    bindingName (Nix.NamedVar (StaticKey name :| []) _ _) = Just (varNameText name)
    bindingName _ = Nothing


-- | Get type from an atom
atomType :: NAtom -> NixType
atomType = \case
  NInt _ -> TInt
  NFloat _ -> TFloat
  NBool _ -> TBool
  NNull -> TNull

-- | Extract text from VarName
varNameText :: VarName -> Text
varNameText = coerce

-- ============================================================================
-- Results
-- ============================================================================

-- | A typed binding
data Binding = Binding
  { bindName :: !Text
  , bindType :: !NixType
  , bindSpan :: !Span
  }
  deriving (Eq, Show)

-- | Inference result for a file
data InferResult = InferResult
  { irBindings :: ![Binding]
  , irFunctions :: ![(Text, NixType)]
  }
  deriving (Eq, Show)

-- | Infer types for a Nix expression
inferExpr :: NExprLoc -> Either Text (NixType, [Binding])
inferExpr expr = Right $ runInfer $ do
  t <- infer builtinEnv expr
  t' <- applyCurrentSubst t
  -- TODO: collect bindings during inference
  pure (t', [])

-- | Infer types for a Nix file
inferFile :: FilePath -> IO (Either Text InferResult)
inferFile path = do
  result <- parseNixFileLoc (Nix.Path path)
  case result of
    Left doc -> pure $ Left (T.pack $ show doc)
    Right expr -> 
      case inferExpr expr of
        Left err -> pure $ Left err
        Right (t, bindings) -> pure $ Right $ InferResult bindings [(T.pack path, t)]
