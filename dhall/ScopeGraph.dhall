{- ScopeGraph.dhall

   Scope graph schema for language-independent name resolution.
   
   Based on: Néron, Tolmach, Visser, Wachsmuth. "A Theory of Name Resolution"
   ESOP 2015. https://doi.org/10.1007/978-3-662-46669-8_9
   
   This is THE interchange format for:
     - render (Nix analysis) -> zeitschrift (documentation)
     - render -> LSP (go-to-definition, find references)
     - render -> armitage (build graph extraction)
   
   If it resolves in the scope graph, it's correct.
   If it doesn't resolve, the build fails.
-}

-- Source position (1-indexed line/column)
let SourcePos
    : Type
    = { line : Natural
      , col : Natural
      }

-- Source span (start to end position, optional file)
let SourceSpan
    : Type
    = { start : SourcePos
      , end : SourcePos
      , file : Optional Text
      }

-- Scope identifier (unique within a graph)
let ScopeId
    : Type
    = Natural

-- What kind of scope this is (for documentation/UI)
let ScopeKind
    : Type
    = < FileScope          -- Top-level file
      | LetScope           -- let ... in
      | AttrSetScope       -- { ... }
      | RecAttrSetScope    -- rec { ... }
      | FunctionScope      -- { args }: body
      | WithScope          -- with expr;
      | ModuleScope        -- Nix module (flake-parts, NixOS, etc.)
      | ClassScope         -- For languages with classes
      | NamespaceScope     -- For languages with namespaces
      >

-- Edge labels determine resolution order
-- Lower priority number = checked first = shadows others
let EdgeLabel
    : Type
    = < Parent             -- Lexical parent scope (priority 1)
      | Import             -- import ./file.nix (priority 3)
      | With               -- with expr; (priority 5, lowest)
      | Inherit            -- inherit (expr) names;
      | AttrAccess         -- x.y (enter x's associated scope)
      | Export             -- Module exports
      | Extend             -- Class inheritance
      | Include            -- C-style include
      | Section            -- Document section containment
      | Citation           -- Bibliography citation
      | Definition         -- Points to definition scope (priority 0)
      >

-- What kind of reference this is
let RefKind
    : Type
    = < VarRef             -- Simple variable: x
      | AttrRef            -- Attribute access: x.y
      | InheritRef         -- inherit x; or inherit (e) x;
      | ImportRef          -- import ./path
      | QualifiedRef       -- Foo.bar (qualified name)
      >

-- A declaration introduces a name into a scope
let Declaration
    : Type
    = { name : Text                    -- The name being declared
      , span : SourceSpan              -- Where it's declared
      , scope : ScopeId                -- Which scope it's in
      , assocScope : Optional ScopeId  -- Scope to enter for qualified access
      , type : Optional Text           -- Inferred type (if available)
      , doc : Optional Text            -- Documentation (if any)
      , kind : Optional Text           -- "function", "variable", "module", etc.
      }

-- A reference uses a name, to be resolved to a declaration
let Reference
    : Type
    = { name : Text                    -- The name being referenced
      , span : SourceSpan              -- Where it's used
      , scope : ScopeId                -- Which scope the reference is in
      , kind : RefKind                 -- What kind of reference
      }

-- An edge connects scopes
let Edge
    : Type
    = { source : ScopeId
      , target : ScopeId
      , label : EdgeLabel
      }

-- Import filter for selective imports
let ImportFilter
    : Type
    = < ImportAll                      -- All declarations visible
      | ImportOnly : List Text         -- Only these names
      | ImportExcept : List Text       -- All except these names
      | ImportAs : { original : Text, alias : Text }  -- Rename
      | ImportQualified : Text         -- Only as Qualifier.name
      >

-- A scope contains declarations and references
let Scope
    : Type
    = { id : ScopeId
      , declarations : List Declaration
      , references : List Reference
      , edges : List Edge
      , kind : ScopeKind
      }

-- Resolution path step
let PathStep
    : Type
    = < StepEdge : { label : EdgeLabel, target : ScopeId }
      | StepDecl : Declaration
      >

-- A resolution path from reference to declaration
let ResolutionPath
    : Type
    = { steps : List PathStep
      , declaration : Declaration
      }

-- Resolution error
let ResolutionError
    : Type
    = < Unresolved : Reference
      | Ambiguous : { reference : Reference, candidates : List Declaration }
      | CycleDetected : { reference : Reference, cycle : List ScopeId }
      >

-- A resolved reference (for the resolution map)
let ResolvedRef
    : Type
    = { reference : Reference
      , declaration : Declaration
      , path : ResolutionPath
      }

-- The complete scope graph
let ScopeGraph
    : Type
    = { scopes : List Scope            -- All scopes
      , root : ScopeId                 -- Entry point scope
      , file : Optional Text           -- Source file (if single-file)
      , files : List Text              -- All files (if multi-file)
      , resolutions : List ResolvedRef -- Pre-computed resolutions
      , errors : List ResolutionError  -- Unresolved references
      }

-- Flake metadata (the contract between render and armitage)
let FlakeMetadata
    : Type
    = { inputs : List { mapKey : Text, mapValue : Text }  -- Flake inputs (name -> flakeref)
      , modules : List { path : Text, class : Text }      -- Modules with their _class
      , packages : List Text                               -- Package names
      , overlays : List Text                               -- Overlay names
      , checks : List Text                                 -- Check names
      , lib : List Text                                    -- Library function names
      , graph : ScopeGraph                                 -- The scope graph
      , violations : List Text                             -- Must be empty for valid flake
      }

-- Constructors
let mkPos
    : Natural -> Natural -> SourcePos
    = \(line : Natural) -> \(col : Natural) ->
        { line, col }

let mkSpan
    : SourcePos -> SourcePos -> Optional Text -> SourceSpan
    = \(start : SourcePos) -> \(end : SourcePos) -> \(file : Optional Text) ->
        { start, end, file }

let mkDecl
    : Text -> SourceSpan -> ScopeId -> Declaration
    = \(name : Text) -> \(span : SourceSpan) -> \(scope : ScopeId) ->
        { name
        , span
        , scope
        , assocScope = None ScopeId
        , type = None Text
        , doc = None Text
        , kind = None Text
        }

let mkRef
    : Text -> SourceSpan -> ScopeId -> RefKind -> Reference
    = \(name : Text) -> \(span : SourceSpan) -> \(scope : ScopeId) -> \(kind : RefKind) ->
        { name, span, scope, kind }

let mkEdge
    : ScopeId -> ScopeId -> EdgeLabel -> Edge
    = \(source : ScopeId) -> \(target : ScopeId) -> \(label : EdgeLabel) ->
        { source, target, label }

let mkScope
    : ScopeId -> ScopeKind -> Scope
    = \(id : ScopeId) -> \(kind : ScopeKind) ->
        { id
        , declarations = [] : List Declaration
        , references = [] : List Reference
        , edges = [] : List Edge
        , kind
        }

let empty
    : ScopeGraph
    = { scopes = [ mkScope 0 ScopeKind.FileScope ]
      , root = 0
      , file = None Text
      , files = [] : List Text
      , resolutions = [] : List ResolvedRef
      , errors = [] : List ResolutionError
      }

in  { -- Types
      SourcePos
    , SourceSpan
    , ScopeId
    , ScopeKind
    , EdgeLabel
    , RefKind
    , Declaration
    , Reference
    , Edge
    , ImportFilter
    , Scope
    , PathStep
    , ResolutionPath
    , ResolutionError
    , ResolvedRef
    , ScopeGraph
    , FlakeMetadata
    
    -- Constructors
    , mkPos
    , mkSpan
    , mkDecl
    , mkRef
    , mkEdge
    , mkScope
    , empty
    }
