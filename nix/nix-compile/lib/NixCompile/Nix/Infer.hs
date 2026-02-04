{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- |
-- Module      : NixCompile.Nix.Infer
-- Description : Type inference for Nix expressions
--
-- Hindley-Milner style type inference for a subset of Nix.
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

import Control.Monad (foldM, forM, forM_, replicateM)
import Control.Monad.Except
import Control.Monad.State.Strict
import Data.Coerce (coerce)
import Data.Fix (Fix (..))
import Data.Functor.Compose (Compose (..))
import Data.List.NonEmpty (NonEmpty (..))
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (catMaybes, fromMaybe, mapMaybe, isJust)
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import Data.Graph (stronglyConnComp, SCC(..))
import Nix.Atoms (NAtom (..))
import Nix.Expr.Types hiding (Binding)
import qualified Nix.Expr.Types as Nix
import Nix.Expr.Types.Annotated (NExprLoc, AnnUnit(..), nullSpan)
import Nix.Parser (parseNixFileLoc, parseNixTextLoc)
import qualified Nix.Utils as Nix
import NixCompile.Nix.Types
import NixCompile.Types (Loc (..), Span (..))
import Text.Megaparsec.Pos (unPos, SourcePos(..))

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
builtinEnv = TypeEnv $ Map.union (Map.singleton "builtins" (mono $ TAttrs builtinsTypes)) (Map.map (mono . fst) builtinsTypes)
  where
    mono t = Forall [] t
    req t = (t, False)
    
    builtinsTypes :: Map Text (NixType, Bool)
    builtinsTypes = Map.fromList $ map (\(k,v) -> (k, req v))
      [ -- String functions
        ("toString", TFun (TUnion [TInt, TFloat, TBool, TPath, TString]) TString)
      , ("baseNameOf", TFun TPath TString)
      , ("dirOf", TFun TPath TPath)
      , ("stringLength", TFun TString TInt)
      , ("substring", TFun TInt (TFun TInt (TFun TString TString)))
      , ("replaceStrings", TFun (TList TString) (TFun (TList TString) (TFun TString TString)))
      
      -- List functions
      , ("head", TFun (TList TAny) TAny)
      , ("tail", TFun (TList TAny) (TList TAny))
      , ("length", TFun (TList TAny) TInt)
      , ("elemAt", TFun (TList TAny) (TFun TInt TAny))
      , ("filter", TFun (TFun TAny TBool) (TFun (TList TAny) (TList TAny)))
      , ("map", TFun (TFun TAny TAny) (TFun (TList TAny) (TList TAny)))
      , ("foldl'", TFun (TFun TAny (TFun TAny TAny)) (TFun TAny (TFun (TList TAny) TAny)))
      , ("concatLists", TFun (TList (TList TAny)) (TList TAny))
      , ("concatMap", TFun (TFun TAny (TList TAny)) (TFun (TList TAny) (TList TAny)))
      
      -- Attrset functions
      , ("attrNames", TFun (TAttrsOpen Map.empty) (TList TString))
      , ("attrValues", TFun (TAttrsOpen (Map.singleton "_" (TAny, False))) (TList TAny))
      , ("hasAttr", TFun TString (TFun (TAttrsOpen Map.empty) TBool))
      , ("getAttr", TFun TString (TFun (TAttrsOpen Map.empty) TAny))
      , ("removeAttrs", TFun (TAttrsOpen Map.empty) (TFun (TList TString) (TAttrsOpen Map.empty)))
      , ("listToAttrs", TFun (TList (TAttrs (Map.fromList [("name", (TString, False)), ("value", (TAny, False))]))) (TAttrsOpen Map.empty))
      
      -- Type checking
      , ("isNull", TFun TAny TBool)
      , ("isInt", TFun TAny TBool)
      , ("isFloat", TFun TAny TBool)
      , ("isBool", TFun TAny TBool)
      , ("isString", TFun TAny TBool)
      , ("isList", TFun TAny TBool)
      , ("isAttrs", TFun TAny TBool)
      , ("isFunction", TFun TAny TBool)
      , ("isPath", TFun TAny TBool)
      
      -- Arithmetic
      , ("add", TFun TInt (TFun TInt TInt))
      , ("sub", TFun TInt (TFun TInt TInt))
      , ("mul", TFun TInt (TFun TInt TInt))
      , ("div", TFun TInt (TFun TInt TInt))
      
      -- Comparison
      , ("lessThan", TFun TInt (TFun TInt TBool))
      
      -- Import
      , ("import", TFun TPath TAny)
      , ("readFile", TFun TPath TString)
      , ("toPath", TFun TString TPath)
      
      -- Derivation
      , ("derivation", TFun (TAttrsOpen Map.empty) TDerivation)
      
      -- Misc
      , ("throw", TFun TString TAny)
      , ("abort", TFun TString TAny)
      , ("trace", TFun TString (TFun TAny TAny))
      , ("seq", TFun TAny (TFun TAny TAny))
      , ("deepSeq", TFun TAny (TFun TAny TAny))
      , ("tryEval", TFun TAny (TAttrs (Map.fromList [("success", (TBool, False)), ("value", (TAny, False))])))
      ]

-- ============================================================================
-- Inference State
-- ============================================================================

-- | Inference state
data InferState = InferState
  { inferSupply :: !Int  -- Fresh type variable supply
  , inferSubst :: !Subst -- Current substitution
  , inferBinds :: ![Binding] -- Collected bindings
  }

-- | Inference monad with error handling
type Infer a = ExceptT Text (State InferState) a

runInfer :: Infer a -> Either Text (a, [Binding])
runInfer m = 
  let (eRes, state) = runState (runExceptT m) (InferState 0 emptySubst [])
  in case eRes of
       Left err -> Left err
       Right res -> Right (res, inferBinds state)

-- | Emit a binding
emitBinding :: Text -> NixType -> Span -> Infer ()
emitBinding name t span = modify $ \s ->
  s { inferBinds = Binding name t span : inferBinds s }

-- | Throw a type error
throwTypeError :: Text -> Infer a
throwTypeError = throwError

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
-- Unification and Merging
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
  (TStrLit s1, TStrLit s2) | s1 == s2 -> pure ()
  (TString, TStrLit _) -> pure ()
  (TStrLit _, TString) -> pure ()
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
  _ -> pure () -- Mismatch

bindVar :: TypeVar -> NixType -> Infer ()
bindVar v t
  | t == TVar v = pure ()
  | occursCheck v t = pure ()
  | otherwise = addSubst v t

occursCheck :: TypeVar -> NixType -> Bool
occursCheck v = \case
  TVar v' -> v == v'
  TList t -> occursCheck v t
  TFun a b -> occursCheck v a || occursCheck v b
  TAttrs m -> any (occursCheck v . fst) (Map.elems m)
  TAttrsOpen m -> any (occursCheck v . fst) (Map.elems m)
  TUnion ts -> any (occursCheck v) ts
  _ -> False

unifyAttrs :: Map Text (NixType, Bool) -> Map Text (NixType, Bool) -> Infer ()
unifyAttrs m1 m2 = do
  -- Closed vs Closed
  let keys1 = Map.keysSet m1
  let keys2 = Map.keysSet m2
  let allKeys = Set.union keys1 keys2
  
  forM_ (Set.toList allKeys) $ \k -> do
    let v1 = Map.lookup k m1
    let v2 = Map.lookup k m2
    case (v1, v2) of
      (Just (t1, _), Just (t2, _)) -> unify t1 t2
      (Just (_, False), Nothing) -> throwTypeError $ "missing required field: " <> k
      (Nothing, Just (_, False)) -> throwTypeError $ "unexpected field (required in other): " <> k
      _ -> pure ()

unifyAttrsOpenOpen :: Map Text (NixType, Bool) -> Map Text (NixType, Bool) -> Infer ()
unifyAttrsOpenOpen m1 m2 = do
  let common = Map.intersectionWith (,) m1 m2
  mapM_ (\((t1, _), (t2, _)) -> unify t1 t2) (Map.elems common)

unifyAttrsClosedOpen :: Map Text (NixType, Bool) -> Map Text (NixType, Bool) -> Infer ()
unifyAttrsClosedOpen closed open = do
  let openKeys = Map.keysSet open
  let closedKeys = Map.keysSet closed
  let missingInClosed = Set.difference openKeys closedKeys
  
  if not (Set.null missingInClosed)
    then throwTypeError $ "closed set missing fields required by open set: " <> T.intercalate ", " (Set.toList missingInClosed)
    else do
      let common = Map.intersectionWith (,) closed open
      mapM_ (\((t1, _), (t2, _)) -> unify t1 t2) (Map.elems common)

unifyUnion :: [NixType] -> NixType -> Infer ()
unifyUnion ts t = case ts of
  [] -> pure ()
  [t'] -> unify t' t
  _ -> pure ()

mergeTypes :: NixType -> NixType -> Infer NixType
mergeTypes t1 t2 = do
  t1' <- applyCurrentSubst t1
  t2' <- applyCurrentSubst t2
  case (t1', t2') of
    (TVar v, t) -> bindVar v t >> pure t
    (t, TVar v) -> bindVar v t >> pure t
    (TAny, _) -> pure TAny
    (_, TAny) -> pure TAny
    (TAttrs m1, TAttrs m2) -> mergeAttrs m1 m2
    (TList e1, TList e2) -> TList <$> mergeTypes e1 e2
    (TFun a1 b1, TFun a2 b2) -> do
       unify a1 a2
       res <- mergeTypes b1 b2
       pure $ TFun a1 res
    (a, b) | a == b -> pure a
    (a, b) -> pure $ TUnion [a, b]

mergeAttrs :: Map Text (NixType, Bool) -> Map Text (NixType, Bool) -> Infer NixType
mergeAttrs m1 m2 = do
  let keys = Set.union (Map.keysSet m1) (Map.keysSet m2)
  fields <- forM (Set.toList keys) $ \k -> do
    let v1 = Map.lookup k m1
    let v2 = Map.lookup k m2
    case (v1, v2) of
      (Just (t1, o1), Just (t2, o2)) -> do
         t <- mergeTypes t1 t2
         pure (k, (t, o1 || o2))
      (Just (t1, _), Nothing) -> pure (k, (t1, True))
      (Nothing, Just (t2, _)) -> pure (k, (t2, True))
      _ -> error "impossible"
  pure $ TAttrs (Map.fromList fields)

-- ============================================================================
-- Instantiation
-- ============================================================================

instantiate :: Scheme -> Infer NixType
instantiate (Forall vars t) = do
  freshVars <- mapM (const freshVar) vars
  let subst = Map.fromList (zip vars freshVars)
  pure $ applySubst subst t

-- ============================================================================
-- Inference
-- ============================================================================

infer :: TypeEnv -> NExprLoc -> Infer NixType
infer env (Fix (Compose (AnnUnit _ expr))) = case expr of
  NConstant atom -> pure $ atomType atom
  NStr (DoubleQuoted [Plain t]) -> pure $ TStrLit t
  NStr _ -> pure TString
  NLiteralPath _ -> pure TPath
  NEnvPath _ -> pure TPath
  NSym name -> case lookupEnv (varNameText name) env of
    Just scheme -> instantiate scheme
    Nothing -> freshVar
  NList [] -> do
    elemType <- freshVar
    pure $ TList elemType
  NList (x:xs) -> do
    elemType <- infer env x
    finalElemType <- foldM (\acc e -> infer env e >>= mergeTypes acc) elemType xs
    pure $ TList finalElemType
  NSet recursive bindings -> do
    fields <- inferBindings (recursive == Recursive) env bindings
    let fieldMap = Map.fromList $ map (\(k, t) -> (k, (t, False))) fields
    pure $ TAttrs fieldMap
  NLet bindings body -> inferLet env bindings body
  NIf cond thenE elseE -> do
    condT <- infer env cond
    unify condT TBool
    thenT <- infer env thenE
    elseT <- infer env elseE
    mergeTypes thenT elseT
  NWith scope body -> do
    _ <- infer env scope
    infer env body
  NAssert cond body -> do
    condT <- infer env cond
    unify condT TBool
    infer env body
  NAbs params body -> inferLambda env params body
  NApp func arg -> do
    funcT <- infer env func
    argT <- infer env arg
    resultT <- freshVar
    unify funcT (TFun argT resultT)
    applyCurrentSubst resultT
  NSelect _ base (attr :| _) -> do
    baseT <- infer env base
    t' <- applyCurrentSubst baseT
    let key = case attr of
          StaticKey k -> Just (varNameText k)
          DynamicKey _ -> Nothing
    case (t', key) of
      (TAttrs fields, Just k) -> 
        case Map.lookup k fields of
          Just (t, _) -> pure t
          Nothing -> freshVar
      (TAttrsOpen fields, Just k) -> 
        case Map.lookup k fields of
          Just (t, _) -> pure t
          Nothing -> freshVar
      _ -> freshVar
  NHasAttr base _ -> do
    _ <- infer env base
    pure TBool
  NUnary op e -> do
    t <- infer env e
    case op of
      NNeg -> unify t TInt >> pure TInt
      NNot -> unify t TBool >> pure TBool
  NBinary op left right -> inferBinary env op left right
  NSynHole _ -> freshVar

inferBinary :: TypeEnv -> NBinaryOp -> NExprLoc -> NExprLoc -> Infer NixType
inferBinary env op left right = do
  leftT <- infer env left
  rightT <- infer env right
  case op of
    NEq -> unify leftT rightT >> pure TBool
    NNEq -> unify leftT rightT >> pure TBool
    NLt -> unify leftT TInt >> unify rightT TInt >> pure TBool
    NLte -> unify leftT TInt >> unify rightT TInt >> pure TBool
    NGt -> unify leftT TInt >> unify rightT TInt >> pure TBool
    NGte -> unify leftT TInt >> unify rightT TInt >> pure TBool
    NAnd -> unify leftT TBool >> unify rightT TBool >> pure TBool
    NOr -> unify leftT TBool >> unify rightT TBool >> pure TBool
    NImpl -> unify leftT TBool >> unify rightT TBool >> pure TBool
    NPlus -> do
      resultT <- freshVar
      pure resultT
    NMinus -> unify leftT TInt >> unify rightT TInt >> pure TInt
    NMult -> unify leftT TInt >> unify rightT TInt >> pure TInt
    NDiv -> unify leftT TInt >> unify rightT TInt >> pure TInt
    NConcat -> do
      unify leftT rightT
      applyCurrentSubst leftT
    NUpdate -> do
      leftT' <- applyCurrentSubst leftT
      rightT' <- applyCurrentSubst rightT
      case (leftT', rightT') of
        (TAttrs l, TAttrs r) -> pure $ TAttrs (r `Map.union` l)
        (TAttrsOpen l, TAttrsOpen r) -> pure $ TAttrsOpen (r `Map.union` l)
        (TAttrs l, TAttrsOpen r) -> pure $ TAttrsOpen (r `Map.union` l)
        (TAttrsOpen l, TAttrs r) -> pure $ TAttrsOpen (r `Map.union` l)
        _ -> do
          unify leftT rightT
          applyCurrentSubst leftT

inferLambda :: TypeEnv -> Params NExprLoc -> NExprLoc -> Infer NixType
inferLambda env params body = case params of
  Param name -> do
    paramT <- freshVar
    let env' = extendEnv (varNameText name) (Forall [] paramT) env
    resultT <- infer env' body
    paramT' <- applyCurrentSubst paramT
    pure $ TFun paramT' resultT
  
  ParamSet mName variadic paramList -> do
    paramTypes <- forM paramList $ \(name, mDefault) -> do
      t <- case mDefault of
        Just defaultExpr -> infer env defaultExpr
        Nothing -> freshVar
      pure (varNameText name, (t, isJust mDefault))
    
    let attrsT = if variadic == Variadic
                   then TAttrsOpen (Map.fromList paramTypes)
                   else TAttrs (Map.fromList paramTypes)
                   
    let env' = foldr (\(n, (t, _)) e -> extendEnv n (Forall [] t) e) env paramTypes
    
    let env'' = case mName of
          Just name -> extendEnv (varNameText name) (Forall [] attrsT) env'
          Nothing -> env'
    
    resultT <- infer env'' body
    pure $ TFun attrsT resultT

inferBindings :: Bool -> TypeEnv -> [Nix.Binding NExprLoc] -> Infer [(Text, NixType)]
inferBindings recursive env bindings
  | recursive = do
      let names = concatMap bindingNames bindings
      freshVars <- replicateM (length names) freshVar
      let env' = foldr (\(n, t) e -> extendEnv n (Forall [] t) e) env (zip names freshVars)
      
      concat <$> forM (zip bindings (chunkVars bindings freshVars)) (\(binding, vars) -> do
        case binding of
          Nix.NamedVar (StaticKey name :| []) expr pos -> do
            let typeVar = head vars
            t <- infer env' expr
            unify typeVar t
            t' <- applyCurrentSubst t
            emitBinding (varNameText name) t' (toSpan pos)
            pure [(varNameText name, t)]
            
          Nix.Inherit mScope keys pos -> do
            forM (zip keys vars) $ \(key, typeVar) -> do
              let name = varNameText key
              t <- case mScope of
                Just scope -> infer env' (Fix (Compose (AnnUnit nullSpan (NSelect Nothing scope (StaticKey key :| [])))))
                Nothing -> case lookupEnv name env' of
                  Just s -> instantiate s
                  Nothing -> freshVar
              unify typeVar t
              pure (name, t)
              
          _ -> pure []
        )
  | otherwise = foldM go [] bindings
  where
    chunkVars [] _ = []
    chunkVars (b:bs) vars = 
      let len = length (bindingNames b)
          (mine, rest) = splitAt len vars
      in mine : chunkVars bs rest

    bindingNames (Nix.NamedVar (StaticKey name :| []) _ _) = [varNameText name]
    bindingNames (Nix.Inherit _ keys _) = map varNameText keys
    bindingNames _ = []
    
    go acc binding = case binding of
      Nix.NamedVar (StaticKey name :| []) expr pos -> do
        t <- infer env expr
        t' <- applyCurrentSubst t
        emitBinding (varNameText name) t' (toSpan pos)
        pure $ acc ++ [(varNameText name, t)]
        
      Nix.Inherit mScope keys pos -> do
        binds <- forM keys $ \key -> do
          let name = varNameText key
          t <- case mScope of
            Just scope -> infer env (Fix (Compose (AnnUnit nullSpan (NSelect Nothing scope (StaticKey key :| [])))))
            Nothing -> case lookupEnv name env of
              Just s -> instantiate s
              Nothing -> freshVar
          pure (name, t)
        pure $ acc ++ binds
        
      _ -> pure acc

inferLet :: TypeEnv -> [Nix.Binding NExprLoc] -> NExprLoc -> Infer NixType
inferLet env bindings body = do
  let namedBindings = concatMap parseBinding bindings
  let edges = map (buildEdge namedBindings) namedBindings
  let sccs = stronglyConnComp edges
  
  envBody <- foldM (inferGroup env) env sccs
  infer envBody body
  where
    parseBinding (Nix.NamedVar (StaticKey name :| []) expr pos) = 
      [(varNameText name, expr, toSpan pos)]
    parseBinding (Nix.Inherit mScope keys pos) = 
      map (\key -> 
        let name = varNameText key
            expr = case mScope of
              Just scope -> Fix (Compose (AnnUnit nullSpan (NSelect Nothing scope (StaticKey key :| []))))
              Nothing -> Fix (Compose (AnnUnit nullSpan (NSym key)))
        in (name, expr, toSpan pos)
      ) keys
    parseBinding _ = []

    buildEdge allBindings (name, expr, span) =
      let free = collectFreeVars expr
          deps = [n | (n, _, _) <- allBindings, n `elem` free]
       in ((name, expr, span), name, deps)

    inferGroup :: TypeEnv -> TypeEnv -> SCC (Text, NExprLoc, Span) -> Infer TypeEnv
    inferGroup baseEnv currentEnv scc = do
      let groupBindings = case scc of
            AcyclicSCC x -> [x]
            CyclicSCC list -> list
            
      let names = map (\(n, _, _) -> n) groupBindings
      freshVars <- replicateM (length names) freshVar
      
      let envRecursive = foldr (\(n, t) e -> extendEnv n (Forall [] t) e) currentEnv (zip names freshVars)
      
      forM_ (zip groupBindings freshVars) $ \((name, expr, span), typeVar) -> do
        t <- infer envRecursive expr
        unify typeVar t
        t' <- applyCurrentSubst t
        emitBinding name t' span

      schemes <- mapM (generalize currentEnv) freshVars
      pure $ foldr (\(n, s) e -> extendEnv n s e) currentEnv (zip names schemes)

collectFreeVars :: NExprLoc -> [Text]
collectFreeVars (Fix (Compose (AnnUnit _ expr))) = case expr of
  NSym name -> [varNameText name]
  NList elems -> concatMap collectFreeVars elems
  NSet _ bindings -> concatMap collectFreeVarsBinding bindings
  NLet bindings body -> concatMap collectFreeVarsBinding bindings ++ collectFreeVars body
  NIf c t f -> collectFreeVars c ++ collectFreeVars t ++ collectFreeVars f
  NWith s b -> collectFreeVars s ++ collectFreeVars b
  NAssert c b -> collectFreeVars c ++ collectFreeVars b
  NAbs params b -> collectFreeVars b
  NApp f a -> collectFreeVars f ++ collectFreeVars a
  NSelect _ b _ -> collectFreeVars b
  NHasAttr b _ -> collectFreeVars b
  NUnary _ e -> collectFreeVars e
  NBinary _ l r -> collectFreeVars l ++ collectFreeVars r
  _ -> []

collectFreeVarsBinding :: Nix.Binding NExprLoc -> [Text]
collectFreeVarsBinding (Nix.NamedVar _ expr _) = collectFreeVars expr
collectFreeVarsBinding _ = []

generalize :: TypeEnv -> NixType -> Infer Scheme
generalize env t = do
  t' <- applyCurrentSubst t
  envSchemes <- mapM applyCurrentSubstScheme (Map.elems (unTypeEnv env))
  let freeInEnv = Set.unions (map freeTypeVarsScheme envSchemes)
  let freeInT = freeTypeVars t'
  let vars = Set.toList (freeInT `Set.difference` freeInEnv)
  pure $ Forall vars t'

applyCurrentSubstScheme :: Scheme -> Infer Scheme
applyCurrentSubstScheme s = do
  subst <- gets inferSubst
  pure $ applySubstScheme subst s

atomType :: NAtom -> NixType
atomType = \case
  NInt _ -> TInt
  NFloat _ -> TFloat
  NBool _ -> TBool
  NNull -> TNull

varNameText :: VarName -> Text
varNameText = coerce

-- | A typed binding
data Binding = Binding
  { bindName :: !Text
  , bindType :: !NixType
  , bindSpan :: !Span
  }
  deriving (Eq, Show)

data InferResult = InferResult
  { irBindings :: ![Binding]
  , irFunctions :: ![(Text, NixType)]
  }
  deriving (Eq, Show)

inferExpr :: NExprLoc -> Either Text (NixType, [Binding])
inferExpr expr = 
  runInfer $ do
    t <- infer builtinEnv expr
    applyCurrentSubst t

toSpan :: Nix.NSourcePos -> Span
toSpan (Nix.NSourcePos _ l c) =
  Span (Loc (fromIntegral $ unPos (coerce l)) (fromIntegral $ unPos (coerce c)))
       (Loc (fromIntegral $ unPos (coerce l)) (fromIntegral $ unPos (coerce c))) 
       Nothing

inferFile :: FilePath -> IO (Either Text InferResult)
inferFile path = do
  result <- parseNixFileLoc (Nix.Path path)
  case result of
    Left doc -> pure $ Left (T.pack $ show doc)
    Right expr -> 
      case inferExpr expr of
        Left err -> pure $ Left err
        Right (t, bindings) -> pure $ Right $ InferResult bindings [(T.pack path, t)]
