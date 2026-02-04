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
import Nix.Expr.Types.Annotated
import Nix.Parser (parseNixFileLoc, parseNixTextLoc)
import qualified Nix.Utils as Nix
import NixCompile.Nix.Types
import NixCompile.Types (Loc (..), Span (..))
import Text.Megaparsec.Pos (unPos)

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
  (TStrLit s1, TStrLit s2) | s1 == s2 -> pure ()
  (TString, TStrLit _) -> pure () -- Subtyping: Literal is a String
  (TStrLit _, TString) -> pure () -- Subtyping: Literal is a String
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
  TAttrs m -> any (occursCheck v . fst) (Map.elems m)
  TAttrsOpen m -> any (occursCheck v . fst) (Map.elems m)
  TUnion ts -> any (occursCheck v) ts
  _ -> False

unifyAttrs :: Map Text (NixType, Bool) -> Map Text (NixType, Bool) -> Infer ()
unifyAttrs m1 m2 = do
  -- Closed vs Closed
  -- Keys must match, unless missing keys are optional in the other map
  let keys1 = Map.keysSet m1
  let keys2 = Map.keysSet m2
  let allKeys = Set.union keys1 keys2
  
  forM_ (Set.toList allKeys) $ \k -> do
    let v1 = Map.lookup k m1
    let v2 = Map.lookup k m2
    case (v1, v2) of
      (Just (t1, _), Just (t2, _)) -> unify t1 t2 -- Both present, unify types
      (Just (_, False), Nothing) -> throwTypeError $ "missing required field: " <> k
      (Nothing, Just (_, False)) -> throwTypeError $ "unexpected field (required in other): " <> k
      _ -> pure () -- Missing but optional, OK

-- | Format a user-friendly row type error (deprecated by custom logic above but kept if needed)
formatRowError :: Map Text NixType -> Map Text NixType -> Set Text -> Set Text -> Text
formatRowError expected actual missing extra = "error"

unifyAttrsOpenOpen :: Map Text (NixType, Bool) -> Map Text (NixType, Bool) -> Infer ()
unifyAttrsOpenOpen m1 m2 = do
  -- Open vs Open: unify common fields
  let common = Map.intersectionWith (,) m1 m2
  mapM_ (\((t1, _), (t2, _)) -> unify t1 t2) (Map.elems common)

unifyAttrsClosedOpen :: Map Text (NixType, Bool) -> Map Text (NixType, Bool) -> Infer ()
unifyAttrsClosedOpen closed open = do
  -- Closed vs Open: 
  -- Open is a subset of Closed? No, Open can have extra fields.
  -- But Closed CANNOT have extra fields.
  -- So keys(Open) must be subset of keys(Closed)? No, Open means "at least these".
  -- Actually TAttrsOpen m means "contains m, and maybe more".
  -- TAttrs m means "contains exactly m".
  -- So TAttrs m ~ TAttrsOpen m'.
  -- m must contain all of m'.
  -- Any key in m' must be in m.
  let openKeys = Map.keysSet open
  let closedKeys = Map.keysSet closed
  let missingInClosed = Set.difference openKeys closedKeys
  
  if not (Set.null missingInClosed)
    then throwTypeError $ "closed set missing fields required by open set: " <> T.intercalate ", " (Set.toList missingInClosed)
    else do
      -- Unify common fields
      let common = Map.intersectionWith (,) closed open
      mapM_ (\((t1, _), (t2, _)) -> unify t1 t2) (Map.elems common)
      -- Check required fields in closed that are missing in open?
      -- If closed requires 'a', and open doesn't mention 'a'.
      -- Open allows 'a'. So it's fine.

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
  NStr (DoubleQuoted [Plain t]) -> pure $ TStrLit t
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
    -- Sets literals are always required fields
    let fieldMap = Map.fromList $ map (\(k, t) -> (k, (t, False))) fields
    pure $ TAttrs fieldMap
  
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
    t' <- applyCurrentSubst baseT
    
    let key = case attr of
          StaticKey k -> Just (varNameText k)
          DynamicKey _ -> Nothing
    
    case (t', key) of
      (TAttrs fields, Just k) -> 
        case Map.lookup k fields of
          Just (t, _) -> pure t
          Nothing -> freshVar -- Missing attr
      (TAttrsOpen fields, Just k) -> 
        case Map.lookup k fields of
          Just (t, _) -> pure t
          Nothing -> freshVar -- Open set
      _ -> freshVar
  
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
    
    -- Attrset update (//) - merges two attrsets, right overrides left
    NUpdate -> do
      leftT' <- applyCurrentSubst leftT
      rightT' <- applyCurrentSubst rightT
      case (leftT', rightT') of
        (TAttrs l, TAttrs r) -> pure $ TAttrs (r `Map.union` l)
        (TAttrsOpen l, TAttrsOpen r) -> pure $ TAttrsOpen (r `Map.union` l)
        (TAttrs l, TAttrsOpen r) -> pure $ TAttrsOpen (r `Map.union` l)
        (TAttrsOpen l, TAttrs r) -> pure $ TAttrsOpen (r `Map.union` l)
        _ -> do
          -- Fallback: try to unify, return left type
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
      pure (varNameText name, (t, isJust mDefault))
    
    -- Pattern types: closed if no ..., open if ... present
    let attrsT = if variadic == Variadic
                   then TAttrsOpen (Map.fromList paramTypes)
                   else TAttrs (Map.fromList paramTypes)
                   
    let env' = foldr (\(n, (t, _)) e -> extendEnv n (Forall [] t) e) env paramTypes
    
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
          Nix.NamedVar (StaticKey name :| []) expr pos -> do
            t <- infer env' expr
            unify typeVar t
            
            -- Emit binding
            t' <- applyCurrentSubst t
            emitBinding (varNameText name) t' (toSpan pos)
            
            pure (varNameText name, t)
          _ -> pure ("", typeVar) -- Skip complex bindings for now
  | otherwise = foldM go [] bindings
  where
    go acc binding = case binding of
      Nix.NamedVar (StaticKey name :| []) expr pos -> do
        t <- infer env expr
        
        -- Emit binding
        t' <- applyCurrentSubst t
        emitBinding (varNameText name) t' (toSpan pos)
        
        pure $ acc ++ [(varNameText name, t)]
      _ -> pure acc

    bindingName (Nix.NamedVar (StaticKey name :| []) _ _) = Just (varNameText name)
    bindingName _ = Nothing

-- | Infer types for a Let block (recursive with generalization)
inferLet :: TypeEnv -> [Nix.Binding NExprLoc] -> NExprLoc -> Infer NixType
inferLet env bindings body = do
  -- 1. Parse bindings into (Name, Expr, Span)
  let namedBindings = mapMaybe parseBinding bindings
      
  -- 2. Build dependency graph
  let edges = map (buildEdge namedBindings) namedBindings
  
  -- 3. SCC analysis
  let sccs = stronglyConnComp edges
  
  -- 4. Process SCCs in order
  envBody <- foldM (inferGroup env) env sccs
  
  -- 5. Infer body
  infer envBody body
  where
    parseBinding (Nix.NamedVar (StaticKey name :| []) expr pos) = 
      Just (varNameText name, expr, toSpan pos)
    parseBinding _ = Nothing

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
      
      -- Extend env with monomorphic variables for recursion within the group
      let envRecursive = foldr (\(n, t) e -> extendEnv n (Forall [] t) e) currentEnv (zip names freshVars)
      
      -- Infer bodies
      forM_ (zip groupBindings freshVars) $ \((name, expr, span), typeVar) -> do
        t <- infer envRecursive expr
        unify typeVar t
        
        -- Emit binding info for IDE/formatting
        t' <- applyCurrentSubst t
        emitBinding name t' span

      -- Generalize
      -- Generalize against currentEnv (variables free in group but not in env are quantified)
      schemes <- mapM (generalize currentEnv) freshVars
      
      -- Extend env with generalized schemes
      pure $ foldr (\(n, s) e -> extendEnv n s e) currentEnv (zip names schemes)

-- | Collect free variables (approximated by all NSym)
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

-- | Generalize a type to a scheme
generalize :: TypeEnv -> NixType -> Infer Scheme
generalize env t = do
  t' <- applyCurrentSubst t
  envSchemes <- mapM applyCurrentSubstScheme (Map.elems (unTypeEnv env))
  let freeInEnv = Set.unions (map freeTypeVarsScheme envSchemes)
  let freeInT = freeTypeVars t'
  let vars = Set.toList (freeInT `Set.difference` freeInEnv)
  pure $ Forall vars t'

-- | Apply substitution to a scheme
applyCurrentSubstScheme :: Scheme -> Infer Scheme
applyCurrentSubstScheme s = do
  subst <- gets inferSubst
  pure $ applySubstScheme subst s

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
inferExpr expr = 
  runInfer $ do
    t <- infer builtinEnv expr
    applyCurrentSubst t

-- | Helper to convert SrcSpan to Span
toSpan :: NSourcePos -> Span
toSpan (NSourcePos _ (NPos l1) (NPos c1)) =
  -- NExprLoc gives start pos. We don't have end pos easily from here without traversing expr.
  -- But we only need start for insertion.
  Span (Loc (fromIntegral $ unPos l1) (fromIntegral $ unPos c1))
       (Loc (fromIntegral $ unPos l1) (fromIntegral $ unPos c1)) 
       Nothing

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
