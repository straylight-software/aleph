{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE StrictData #-}

-- |
-- Module      : NixCompile.Nix.Scope
-- Description : Scope graphs for Nix
--
-- Implements Visser's scope graph theory for Nix expressions.
-- This enables:
--   - Go-to-definition
--   - Find references
--   - Rename refactoring
--   - Documentation generation (zeitschrift integration)
--   - Cross-file analysis
--
-- The scope graph is THE data structure for tooling. Everything else
-- (hover info, completions, diagnostics) derives from it.
--
-- Reference: Néron, Tolmach, Visser, Wachsmuth. "A Theory of Name Resolution"
-- ESOP 2015. https://doi.org/10.1007/978-3-662-46669-8_9
module NixCompile.Nix.Scope
  ( -- * Core Types
    ScopeGraph (..),
    Scope (..),
    ScopeId (..),
    Declaration (..),
    Reference (..),
    Edge (..),
    EdgeLabel (..),
    
    -- * Source Locations
    SourceSpan (..),
    SourcePos (..),
    
    -- * Construction
    empty,
    fromNixExpr,
    fromNixFile,
    fromModuleGraph,
    
    -- * Resolution
    resolve,
    resolveAll,
    ResolutionError (..),
    
    -- * Queries
    declarationsInScope,
    referencesInScope,
    findDeclaration,
    findReferences,
    
    -- * Export (for zeitschrift)
    toJSON,
    toDhall,
  )
where

import Control.Monad (forM_)
import Control.Monad.State.Strict
import Data.Aeson (ToJSON (..), ToJSONKey (..), (.=))
import qualified Data.Aeson as Aeson
import Dhall (ToDhall(..))
import qualified Dhall
import qualified Dhall.Core as Dhall
import qualified Dhall.Marshal.Encode as Encode
import Data.Coerce (coerce)
import Data.Fix (Fix (..))
import Data.Functor.Compose (Compose (..))
import Data.List.NonEmpty (NonEmpty (..))
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import GHC.Generics (Generic)
import Nix.Atoms (NAtom (..))
import Nix.Expr.Types hiding (Binding, SourcePos)
import qualified Nix.Expr.Types as Nix
import Nix.Expr.Types.Annotated
import Nix.Utils (Path(..))
import Text.Megaparsec.Pos (unPos)
import Numeric.Natural (Natural)

-- ============================================================================
-- Core Types
-- ============================================================================

-- | A scope graph for Nix code.
--
-- Note: For Dhall export, we use a separate 'ScopeGraphExport' type
-- that uses Natural instead of Int and List instead of Map, to match
-- the Dhall schema in dhall/ScopeGraph.dhall.
data ScopeGraph = ScopeGraph
  { sgScopes :: Map ScopeId Scope
  , sgRoot :: ScopeId
  , sgNextId :: Int
  , sgFile :: Maybe FilePath
  }
  deriving stock (Eq, Show, Generic)

-- | A scope contains declarations and references.
data Scope = Scope
  { scopeId :: ScopeId
  , scopeDeclarations :: [Declaration]
  , scopeReferences :: [Reference]
  , scopeEdges :: [Edge]
  , scopeKind :: ScopeKind
  }
  deriving stock (Eq, Show, Generic)

-- | What kind of scope this is (for documentation/UI).
data ScopeKind
  = FileScope          -- ^ Top-level file
  | LetScope           -- ^ let ... in
  | AttrSetScope       -- ^ { ... }
  | RecAttrSetScope    -- ^ rec { ... }
  | FunctionScope      -- ^ { args }: body
  | WithScope          -- ^ with expr;
  deriving stock (Eq, Show, Generic)
  deriving anyclass (ToDhall)

newtype ScopeId = ScopeId { unScopeId :: Int }
  deriving stock (Eq, Ord, Show, Generic)
  deriving newtype (Num, ToJSON, ToJSONKey)

-- | A declaration introduces a name into a scope.
data Declaration = Declaration
  { declName :: Text
  , declSpan :: SourceSpan
  , declScope :: ScopeId
  , declAssocScope :: Maybe ScopeId  -- ^ For attr sets: scope of the value
  , declType :: Maybe Text           -- ^ Inferred type (if available)
  , declDoc :: Maybe Text            -- ^ Doc comment (if any)
  }
  deriving stock (Eq, Show, Generic)

-- | A reference uses a name, to be resolved to a declaration.
data Reference = Reference
  { refName :: Text
  , refSpan :: SourceSpan
  , refScope :: ScopeId
  , refKind :: RefKind
  }
  deriving stock (Eq, Show, Generic)

-- | What kind of reference this is.
data RefKind
  = VarRef             -- ^ Simple variable: x
  | AttrRef            -- ^ Attribute access: x.y
  | InheritRef         -- ^ inherit x; or inherit (e) x;
  | ImportRef          -- ^ import ./path
  deriving stock (Eq, Show, Generic)
  deriving anyclass (ToDhall)

-- | An edge connects scopes.
data Edge = Edge
  { edgeSource :: ScopeId
  , edgeTarget :: ScopeId
  , edgeLabel :: EdgeLabel
  }
  deriving stock (Eq, Show, Generic)

-- | Edge labels determine resolution order.
data EdgeLabel
  = Parent             -- ^ Lexical parent scope
  | Import             -- ^ import ./file.nix
  | With               -- ^ with expr; (low priority)
  | Inherit            -- ^ inherit (expr) names;
  | AttrAccess         -- ^ x.y (enter x's scope)
  deriving stock (Eq, Ord, Show, Generic)
  deriving anyclass (ToDhall)

-- ============================================================================
-- Source Locations
-- ============================================================================

data SourceSpan = SourceSpan
  { spanStart :: SourcePos
  , spanEnd :: SourcePos
  , spanFile :: Maybe FilePath
  }
  deriving stock (Eq, Show, Generic)

data SourcePos = SourcePos
  { posLine :: Int
  , posCol :: Int
  }
  deriving stock (Eq, Show, Generic)

-- ============================================================================
-- Construction State
-- ============================================================================

data BuildState = BuildState
  { bsGraph :: ScopeGraph
  , bsCurrentScope :: ScopeId
  }

type Build a = State BuildState a

freshScope :: ScopeKind -> Build ScopeId
freshScope kind = do
  st <- get
  let newId = ScopeId (sgNextId (bsGraph st))
  let scope = Scope newId [] [] [] kind
  put st
    { bsGraph = (bsGraph st)
        { sgScopes = Map.insert newId scope (sgScopes (bsGraph st))
        , sgNextId = sgNextId (bsGraph st) + 1
        }
    }
  pure newId

addDecl :: Declaration -> Build ()
addDecl decl = do
  st <- get
  let sid = declScope decl
  let update s = s { scopeDeclarations = decl : scopeDeclarations s }
  put st
    { bsGraph = (bsGraph st)
        { sgScopes = Map.adjust update sid (sgScopes (bsGraph st))
        }
    }

addRef :: Reference -> Build ()
addRef ref = do
  st <- get
  let sid = refScope ref
  let update s = s { scopeReferences = ref : scopeReferences s }
  put st
    { bsGraph = (bsGraph st)
        { sgScopes = Map.adjust update sid (sgScopes (bsGraph st))
        }
    }

addEdge :: Edge -> Build ()
addEdge edge = do
  st <- get
  let sid = edgeSource edge
  let update s = s { scopeEdges = edge : scopeEdges s }
  put st
    { bsGraph = (bsGraph st)
        { sgScopes = Map.adjust update sid (sgScopes (bsGraph st))
        }
    }

currentScope :: Build ScopeId
currentScope = gets bsCurrentScope

withScope :: ScopeId -> Build a -> Build a
withScope sid action = do
  old <- gets bsCurrentScope
  modify $ \st -> st { bsCurrentScope = sid }
  result <- action
  modify $ \st -> st { bsCurrentScope = old }
  pure result

-- ============================================================================
-- Construction from Nix
-- ============================================================================

-- | Create an empty scope graph.
empty :: ScopeGraph
empty = ScopeGraph
  { sgScopes = Map.singleton (ScopeId 0) (Scope (ScopeId 0) [] [] [] FileScope)
  , sgRoot = ScopeId 0
  , sgNextId = 1
  , sgFile = Nothing
  }

-- | Build a scope graph from a Nix expression.
fromNixExpr :: Maybe FilePath -> NExprLoc -> ScopeGraph
fromNixExpr mpath expr =
  let initState = BuildState
        { bsGraph = empty { sgFile = mpath }
        , bsCurrentScope = ScopeId 0
        }
      finalState = execState (buildExpr expr) initState
  in bsGraph finalState

-- | Build a scope graph from a Nix file.
fromNixFile :: FilePath -> NExprLoc -> ScopeGraph
fromNixFile = fromNixExpr . Just

-- | Build scope graph from an already-built module graph.
fromModuleGraph :: Map FilePath NExprLoc -> ScopeGraph
fromModuleGraph modules =
  -- TODO: Build a unified scope graph with import edges between files
  -- For now, just process each file independently
  let graphs = Map.mapWithKey fromNixFile modules
  in case Map.elems graphs of
    [] -> empty
    (g:_) -> g  -- Return first for now

-- | Build scope graph from a Nix expression.
buildExpr :: NExprLoc -> Build ()
buildExpr (Fix (Compose (AnnUnit srcSpan e))) = case e of
  -- Let bindings: introduce a new scope
  NLet bindings body -> do
    parent <- currentScope
    letScope <- freshScope LetScope
    addEdge (Edge letScope parent Parent)
    
    withScope letScope $ do
      -- First pass: add all declarations (for mutual recursion)
      mapM_ (addBindingDecl letScope) bindings
      -- Second pass: process binding values
      mapM_ buildBinding bindings
      -- Process body
      buildExpr body
  
  -- Attribute set: each binding is a declaration
  NSet NonRecursive bindings -> do
    parent <- currentScope
    attrScope <- freshScope AttrSetScope
    addEdge (Edge attrScope parent Parent)
    
    withScope attrScope $ do
      mapM_ (addBindingDecl attrScope) bindings
      mapM_ buildBinding bindings
  
  -- Recursive attribute set
  NSet Recursive bindings -> do
    parent <- currentScope
    attrScope <- freshScope RecAttrSetScope
    addEdge (Edge attrScope parent Parent)
    
    withScope attrScope $ do
      mapM_ (addBindingDecl attrScope) bindings
      mapM_ buildBinding bindings
  
  -- Function: parameters are declarations in function scope
  NAbs params body -> do
    parent <- currentScope
    funScope <- freshScope FunctionScope
    addEdge (Edge funScope parent Parent)
    
    withScope funScope $ do
      addParamDecls funScope params
      buildExpr body
  
  -- With: adds a low-priority scope
  NWith withExpr body -> do
    parent <- currentScope
    withScopeId <- freshScope WithScope
    addEdge (Edge withScopeId parent Parent)
    -- The with expression's scope has low priority
    addEdge (Edge withScopeId parent With)
    
    buildExpr withExpr
    withScope withScopeId $ buildExpr body
  
  -- Variable reference
  NSym name -> do
    scope <- currentScope
    addRef $ Reference
      { refName = coerce name
      , refSpan = toSourceSpan srcSpan
      , refScope = scope
      , refKind = VarRef
      }
  
  -- Attribute selection: x.y
  NSelect _ base (attr :| rest) -> do
    buildExpr base
    -- The attribute path creates references
    scope <- currentScope
    addRef $ Reference
      { refName = keyToText attr
      , refSpan = toSourceSpan srcSpan
      , refScope = scope
      , refKind = AttrRef
      }
    mapM_ (\k -> addRef $ Reference
      { refName = keyToText k
      , refSpan = toSourceSpan srcSpan
      , refScope = scope
      , refKind = AttrRef
      }) rest
  
  -- Application: recurse into both
  NApp f x -> do
    buildExpr f
    buildExpr x
  
  -- Binary op
  NBinary _ l r -> do
    buildExpr l
    buildExpr r
  
  -- Unary op
  NUnary _ x -> buildExpr x
  
  -- If-then-else
  NIf c t f -> do
    buildExpr c
    buildExpr t
    buildExpr f
  
  -- Assert
  NAssert c b -> do
    buildExpr c
    buildExpr b
  
  -- List
  NList xs -> mapM_ buildExpr xs
  
  -- Literals and other cases: no bindings
  _ -> pure ()

-- | Add a binding's declaration to a scope.
addBindingDecl :: ScopeId -> Nix.Binding NExprLoc -> Build ()
addBindingDecl scope = \case
  Nix.NamedVar (StaticKey name :| []) _ srcSpan -> do
    addDecl $ Declaration
      { declName = coerce name
      , declSpan = toSourceSpan' srcSpan
      , declScope = scope
      , declAssocScope = Nothing
      , declType = Nothing
      , declDoc = Nothing
      }
  
  Nix.Inherit Nothing names srcSpan ->
    forM_ names $ \varName ->
      addDecl $ Declaration
        { declName = coerce varName
        , declSpan = toSourceSpan' srcSpan
        , declScope = scope
        , declAssocScope = Nothing
        , declType = Nothing
        , declDoc = Nothing
        }
  
  Nix.Inherit (Just _) names srcSpan ->
    forM_ names $ \varName ->
      addDecl $ Declaration
        { declName = coerce varName
        , declSpan = toSourceSpan' srcSpan
        , declScope = scope
        , declAssocScope = Nothing
        , declType = Nothing
        , declDoc = Nothing
        }
  
  _ -> pure ()

-- | Process a binding's value.
buildBinding :: Nix.Binding NExprLoc -> Build ()
buildBinding = \case
  Nix.NamedVar _ expr _ -> buildExpr expr
  Nix.Inherit (Just expr) _ _ -> buildExpr expr
  Nix.Inherit Nothing _ _ -> pure ()

-- | Add function parameter declarations.
addParamDecls :: ScopeId -> Params NExprLoc -> Build ()
addParamDecls scope = \case
  Param name -> 
    addDecl $ Declaration
      { declName = coerce name
      , declSpan = emptySpan
      , declScope = scope
      , declAssocScope = Nothing
      , declType = Nothing
      , declDoc = Nothing
      }
  
  ParamSet mname _variadic pset -> do
    -- Named parameter set: { ... } @ name
    case mname of
      Just pname -> addDecl $ Declaration
        { declName = coerce pname
        , declSpan = emptySpan
        , declScope = scope
        , declAssocScope = Nothing
        , declType = Nothing
        , declDoc = Nothing
        }
      Nothing -> pure ()
    
    -- Each parameter in the set
    forM_ pset $ \(pname, mdefault) -> do
      addDecl $ Declaration
        { declName = coerce pname
        , declSpan = emptySpan
        , declScope = scope
        , declAssocScope = Nothing
        , declType = Nothing
        , declDoc = Nothing
        }
      -- Process default value if present
      case mdefault of
        Just expr -> buildExpr expr
        Nothing -> pure ()

-- ============================================================================
-- Resolution
-- ============================================================================

data ResolutionError
  = Unresolved Reference
  | Ambiguous Reference [Declaration]
  deriving stock (Eq, Show, Generic)

-- | Resolve a single reference.
resolve :: ScopeGraph -> Reference -> Either ResolutionError Declaration
resolve sg ref = 
  case findPaths sg (refScope ref) (refName ref) of
    [] -> Left (Unresolved ref)
    [d] -> Right d
    ds -> Left (Ambiguous ref ds)

-- | Resolve all references in the scope graph.
resolveAll :: ScopeGraph -> Either [ResolutionError] [(Reference, Declaration)]
resolveAll sg =
  let refs = concatMap scopeReferences (Map.elems (sgScopes sg))
      results = map (\r -> (r, resolve sg r)) refs
      errors = [e | (_, Left e) <- results]
      successes = [(r, d) | (r, Right d) <- results]
  in if null errors
     then Right successes
     else Left errors

-- | Find all declarations reachable with a given name.
findPaths :: ScopeGraph -> ScopeId -> Text -> [Declaration]
findPaths sg startScope name = go Set.empty startScope
  where
    go :: Set ScopeId -> ScopeId -> [Declaration]
    go visited sid
      | Set.member sid visited = []
      | otherwise =
          case Map.lookup sid (sgScopes sg) of
            Nothing -> []
            Just scope ->
              let visited' = Set.insert sid visited
                  -- Local declarations with matching name
                  local = filter (\d -> declName d == name) (scopeDeclarations scope)
                  -- Follow edges (sorted by priority)
                  fromEdges = concatMap (go visited' . edgeTarget) 
                            $ sortEdges (scopeEdges scope)
              in if not (null local) then local else fromEdges
    
    -- Sort edges by priority (Parent > Import > With)
    sortEdges = Prelude.id  -- TODO: implement proper ordering

-- ============================================================================
-- Queries
-- ============================================================================

-- | Get all declarations visible in a scope.
declarationsInScope :: ScopeGraph -> ScopeId -> [Declaration]
declarationsInScope sg sid = go Set.empty sid
  where
    go visited s
      | Set.member s visited = []
      | otherwise = case Map.lookup s (sgScopes sg) of
          Nothing -> []
          Just scope ->
            let visited' = Set.insert s visited
            in scopeDeclarations scope ++ 
               concatMap (go visited' . edgeTarget) (scopeEdges scope)

-- | Get all references in a scope.
referencesInScope :: ScopeGraph -> ScopeId -> [Reference]
referencesInScope sg sid = case Map.lookup sid (sgScopes sg) of
  Nothing -> []
  Just scope -> scopeReferences scope

-- | Find a declaration by name.
findDeclaration :: ScopeGraph -> Text -> [Declaration]
findDeclaration sg name =
  [ d 
  | scope <- Map.elems (sgScopes sg)
  , d <- scopeDeclarations scope
  , declName d == name
  ]

-- | Find all references to a declaration.
findReferences :: ScopeGraph -> Declaration -> [Reference]
findReferences sg decl =
  [ r
  | scope <- Map.elems (sgScopes sg)
  , r <- scopeReferences scope
  , refName r == declName decl
  -- TODO: actually resolve to check it points to this decl
  ]

-- ============================================================================
-- JSON Export (for zeitschrift)
-- ============================================================================

instance ToJSON ScopeGraph where
  toJSON sg = Aeson.object
    [ "scopes" .= sgScopes sg
    , "root" .= sgRoot sg
    , "file" .= sgFile sg
    ]

instance ToJSON Scope where
  toJSON s = Aeson.object
    [ "id" .= scopeId s
    , "declarations" .= scopeDeclarations s
    , "references" .= scopeReferences s
    , "edges" .= scopeEdges s
    , "kind" .= show (scopeKind s)
    ]

instance ToJSON Declaration where
  toJSON d = Aeson.object
    [ "name" .= declName d
    , "span" .= declSpan d
    , "scope" .= declScope d
    , "assocScope" .= declAssocScope d
    , "type" .= declType d
    , "doc" .= declDoc d
    ]

instance ToJSON Reference where
  toJSON r = Aeson.object
    [ "name" .= refName r
    , "span" .= refSpan r
    , "scope" .= refScope r
    , "kind" .= show (refKind r)
    ]

instance ToJSON Edge where
  toJSON e = Aeson.object
    [ "source" .= edgeSource e
    , "target" .= edgeTarget e
    , "label" .= show (edgeLabel e)
    ]

instance ToJSON SourceSpan where
  toJSON s = Aeson.object
    [ "start" .= spanStart s
    , "end" .= spanEnd s
    , "file" .= spanFile s
    ]

instance ToJSON SourcePos where
  toJSON p = Aeson.object
    [ "line" .= posLine p
    , "col" .= posCol p
    ]

-- ============================================================================
-- Dhall Export (for zeitschrift)
-- ============================================================================
--
-- We define separate "export" types that match the Dhall schema exactly.
-- This allows us to use Dhall's generic ToDhall derivation for proper
-- serialization instead of string concatenation.

-- | Source position for Dhall export (uses Natural)
data SourcePosExport = SourcePosExport
  { line :: Natural
  , col :: Natural
  }
  deriving stock (Eq, Show, Generic)
  deriving anyclass (ToDhall)

-- | Source span for Dhall export
data SourceSpanExport = SourceSpanExport
  { start :: SourcePosExport
  , end :: SourcePosExport
  , file :: Maybe Text
  }
  deriving stock (Eq, Show, Generic)
  deriving anyclass (ToDhall)

-- | Declaration for Dhall export
data DeclarationExport = DeclarationExport
  { name :: Text
  , span :: SourceSpanExport
  , scope :: Natural
  , assocScope :: Maybe Natural
  , type_ :: Maybe Text  -- 'type' is reserved, use type_
  , doc :: Maybe Text
  , kind :: Maybe Text
  }
  deriving stock (Eq, Show, Generic)

-- Manual ToDhall instance to rename type_ -> type
instance ToDhall DeclarationExport where
  injectWith _normalizer =
    let opts = Encode.defaultInterpretOptions
          { Encode.fieldModifier = T.dropWhileEnd (== '_') }
    in Encode.genericToDhallWith opts

-- | Reference for Dhall export
data ReferenceExport = ReferenceExport
  { name :: Text
  , span :: SourceSpanExport
  , scope :: Natural
  , kind :: RefKind
  }
  deriving stock (Eq, Show, Generic)
  deriving anyclass (ToDhall)

-- | Edge for Dhall export
data EdgeExport = EdgeExport
  { source :: Natural
  , target :: Natural
  , label :: EdgeLabel
  }
  deriving stock (Eq, Show, Generic)
  deriving anyclass (ToDhall)

-- | Scope for Dhall export
data ScopeExport = ScopeExport
  { id :: Natural
  , declarations :: [DeclarationExport]
  , references :: [ReferenceExport]
  , edges :: [EdgeExport]
  , kind :: ScopeKind
  }
  deriving stock (Eq, Show, Generic)
  deriving anyclass (ToDhall)

-- | Scope graph for Dhall export (matches dhall/ScopeGraph.dhall)
data ScopeGraphExport = ScopeGraphExport
  { scopes :: [ScopeExport]
  , root :: Natural
  , file :: Maybe Text
  , files :: [Text]
  }
  deriving stock (Eq, Show, Generic)
  deriving anyclass (ToDhall)

-- | Convert internal ScopeGraph to export format
toExport :: ScopeGraph -> ScopeGraphExport
toExport sg = ScopeGraphExport
  { scopes = map scopeToExport (Map.elems (sgScopes sg))
  , root = fromIntegral (unScopeId (sgRoot sg))
  , file = T.pack <$> sgFile sg
  , files = []
  }
  where
    scopeToExport :: Scope -> ScopeExport
    scopeToExport s = ScopeExport
      { id = fromIntegral (unScopeId (scopeId s))
      , declarations = map declToExport (scopeDeclarations s)
      , references = map refToExport (scopeReferences s)
      , edges = map edgeToExport (scopeEdges s)
      , kind = scopeKind s
      }
    
    declToExport :: Declaration -> DeclarationExport
    declToExport d = DeclarationExport
      { name = declName d
      , span = spanToExport (declSpan d)
      , scope = fromIntegral (unScopeId (declScope d))
      , assocScope = fromIntegral . unScopeId <$> declAssocScope d
      , type_ = declType d
      , doc = declDoc d
      , kind = Nothing
      }
    
    refToExport :: Reference -> ReferenceExport
    refToExport r = ReferenceExport
      { name = refName r
      , span = spanToExport (refSpan r)
      , scope = fromIntegral (unScopeId (refScope r))
      , kind = refKind r
      }
    
    edgeToExport :: Edge -> EdgeExport
    edgeToExport e = EdgeExport
      { source = fromIntegral (unScopeId (edgeSource e))
      , target = fromIntegral (unScopeId (edgeTarget e))
      , label = edgeLabel e
      }
    
    spanToExport :: SourceSpan -> SourceSpanExport
    spanToExport sp = SourceSpanExport
      { start = posToExport (spanStart sp)
      , end = posToExport (spanEnd sp)
      , file = T.pack <$> spanFile sp
      }
    
    posToExport :: SourcePos -> SourcePosExport
    posToExport p = SourcePosExport
      { line = fromIntegral (posLine p)
      , col = fromIntegral (posCol p)
      }

-- | Emit scope graph as Dhall expression using proper serialization
toDhall :: ScopeGraph -> Text
toDhall sg = Dhall.pretty (Dhall.embed Dhall.inject (toExport sg))

-- ============================================================================
-- Utilities
-- ============================================================================

toSourceSpan :: SrcSpan -> SourceSpan
toSourceSpan srcSpan = SourceSpan
  { spanStart = SourcePos (sourceLine $ getSpanBegin srcSpan) (sourceCol $ getSpanBegin srcSpan)
  , spanEnd = SourcePos (sourceLine $ getSpanEnd srcSpan) (sourceCol $ getSpanEnd srcSpan)
  , spanFile = Nothing
  }
  where
    sourceLine (NSourcePos _ (NPos l) _) = fromIntegral (unPos l)
    sourceCol (NSourcePos _ _ (NPos c)) = fromIntegral (unPos c)

toSourceSpan' :: NSourcePos -> SourceSpan
toSourceSpan' (NSourcePos path (NPos l) (NPos c)) = SourceSpan
  { spanStart = SourcePos (fromIntegral (unPos l)) (fromIntegral (unPos c))
  , spanEnd = SourcePos (fromIntegral (unPos l)) (fromIntegral (unPos c))
  , spanFile = Just (coerce path)
  }

emptySpan :: SourceSpan
emptySpan = SourceSpan (SourcePos 0 0) (SourcePos 0 0) Nothing

keyToText :: NKeyName r -> Text
keyToText (StaticKey name) = coerce name
keyToText (DynamicKey _) = "<dynamic>"
