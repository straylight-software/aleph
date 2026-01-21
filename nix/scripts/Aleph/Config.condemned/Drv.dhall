-- Aleph/Drv.dhall
-- 
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--                           // CA-Derivation Schema //
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--
--     The sky above the port was the color of television, tuned to a dead
--     channel.
--                                                           — Neuromancer
--
-- Sound Dhall schema for content-addressed derivations. Bridges eval and build
-- with typed actions - no regex on [Text], no shell string interpolation.
--
-- Design principles:
--   1. Store paths are typed, not Text
--   2. Build actions are an AST, not shell strings
--   3. References between deps are named, not positional
--   4. CA outputs have explicit hash methods
--
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

-- ============================================================================
-- Hash Types
-- ============================================================================

let HashAlgo = < SHA256 | SHA512 | Blake3 >

let Hash = { algo : HashAlgo, value : Text }

let SriHash = Text  -- "sha256-abc123..." format

-- ============================================================================
-- Store Primitives
-- ============================================================================

-- | A validated store path (must start with /nix/store/)
let StorePath = { _type : Text, path : Text }

let mkStorePath : Text → StorePath =
  λ(p : Text) → { _type = "StorePath", path = p }

-- | A derivation path (ends in .drv)
let DrvPath = { _type : Text, path : Text }

-- | Output reference - either known (Fixed) or computed at build (Floating)
let OutputMethod = < Fixed : SriHash | Floating >

-- ============================================================================
-- References (no string interpolation)
-- ============================================================================

-- | How to reference things in build actions
let Ref = 
  < Dep : { name : Text, subpath : Optional Text }   -- ${deps.name}/subpath
  | Out : { name : Text, subpath : Optional Text }   -- ${outputs.name}/subpath
  | Src : { subpath : Optional Text }                -- ${src}/subpath  
  | Env : Text                                       -- $VAR
  | Rel : Text                                       -- relative path
  | Lit : Text                                       -- literal string
  | Cat : List Ref                                   -- concatenation
  >

-- Smart constructors
let dep = λ(name : Text) → Ref.Dep { name, subpath = None Text }
let depSub = λ(name : Text) → λ(sub : Text) → Ref.Dep { name, subpath = Some sub }
let out = Ref.Out { name = "out", subpath = None Text }
let outSub = λ(sub : Text) → Ref.Out { name = "out", subpath = Some sub }
let outNamed = λ(name : Text) → Ref.Out { name, subpath = None Text }
let src = Ref.Src { subpath = None Text }
let srcSub = λ(sub : Text) → Ref.Src { subpath = Some sub }
let env = Ref.Env
let rel = Ref.Rel
let lit = Ref.Lit
let cat = Ref.Cat

-- ============================================================================
-- Typed Build Actions (the AST)
-- ============================================================================

-- | File permissions
let Mode = < R : {} | RW : {} | RX : {} | RWX : {} | Octal : Natural >

-- | Comparison operators
let Cmp = < Eq | Ne | Lt | Le | Gt | Ge >

-- | Stream targets
let StreamTarget = < Stdout | Stderr | File : Ref | Null >

-- | Expressions (for conditionals and string ops)
let Expr =
  < Str : Text
  | Int : Integer
  | Bool : Bool
  | Ref : Ref
  | Env : Text
  | Concat : List Expr
  | PathExists : Ref
  | FileContents : Ref
  | Compare : { op : Cmp, a : Expr, b : Expr }
  | And : { a : Expr, b : Expr }
  | Or : { a : Expr, b : Expr }
  | Not : Expr
  >

-- | Typed build operations
let Action =
  -- Filesystem
  < Copy : { src : Ref, dst : Ref }
  | Move : { src : Ref, dst : Ref }
  | Symlink : { target : Ref, link : Ref }
  | Mkdir : { path : Ref, parents : Bool }
  | Remove : { path : Ref, recursive : Bool }
  | Touch : Ref
  | Chmod : { path : Ref, mode : Mode }
  
  -- File I/O
  | Write : { path : Ref, contents : Text }
  | Append : { path : Ref, contents : Text }
  
  -- Archives
  | Untar : { src : Ref, dst : Ref, strip : Natural }
  | Unzip : { src : Ref, dst : Ref }
  | Tar : { src : Ref, dst : Ref, compression : < None | Gzip | Zstd | Xz > }
  
  -- Patching
  | Patch : { patch : Ref, dir : Ref, strip : Natural }
  | Substitute : { file : Ref, replacements : List { from : Text, to : Text } }
  | SubstituteRef : { file : Ref, replacements : List { from : Text, to : Ref } }
  
  -- ELF manipulation (patchelf)
  | PatchElfRpath : { path : Ref, rpaths : List Ref }
  | PatchElfAddRpath : { path : Ref, rpaths : List Ref }
  | PatchElfInterpreter : { path : Ref, interpreter : Ref }
  | PatchElfShrink : { path : Ref }
  
  -- Program execution
  | Run : { cmd : Ref
          , args : List Expr
          , env : List { key : Text, value : Expr }
          , cwd : Optional Ref
          , stdin : Optional Ref
          , stdout : StreamTarget
          , stderr : StreamTarget
          }
  
  -- Tool invocations (resolved from deps)
  | Tool : { dep : Text           -- dep name
           , bin : Text           -- binary name within dep
           , args : List Expr
           }
  
  -- Build systems
  | CMake : { srcDir : Ref
            , buildDir : Ref
            , installPrefix : Ref
            , buildType : Text
            , flags : List Text
            , generator : < Ninja | Make | Default >
            }
  | CMakeBuild : { buildDir : Ref, target : Optional Text, jobs : Optional Natural }
  | CMakeInstall : { buildDir : Ref }
  
  | Make : { targets : List Text
           , flags : List Text
           , jobs : Optional Natural
           , dir : Optional Ref
           }
  
  | Meson : { srcDir : Ref
            , buildDir : Ref
            , prefix : Ref
            , buildType : Text
            , flags : List Text
            }
  | NinjaBuild : { buildDir : Ref, targets : List Text, jobs : Optional Natural }
  
  | Configure : { flags : List Text }
  
  -- Install helpers
  | InstallBin : { src : Ref }
  | InstallLib : { src : Ref }
  | InstallInclude : { src : Ref }
  | InstallShare : { src : Ref, subdir : Text }
  | InstallPkgConfig : { src : Ref }
  
  -- Control flow
  | If : { cond : Expr, then_ : List Action, else_ : List Action }
  | ForFiles : { pattern : Text, dir : Ref, var : Text, do : List Action }
  | Seq : List Action
  | Parallel : List Action
  | Try : { actions : List Action, catch : List Action }
  
  -- Assertions
  | Assert : { cond : Expr, msg : Text }
  | Log : { level : < Debug | Info | Warn | Error >, msg : Text }
  
  -- Escape hatch
  | Shell : Text
  >

-- ============================================================================
-- Dependencies
-- ============================================================================

let DepKind = < Build | Host | Propagate | Check | Data >

let Dep = 
  { name : Text
  , storePath : Optional StorePath  -- None means resolve at build time
  , kind : DepKind
  , outputs : List Text             -- which outputs we need
  }

let mkDep : Text → DepKind → Dep =
  λ(name : Text) → λ(kind : DepKind) → 
    { name
    , storePath = None StorePath
    , kind
    , outputs = ["out"]
    }

let buildDep = λ(name : Text) → mkDep name DepKind.Build
let hostDep = λ(name : Text) → mkDep name DepKind.Host
let checkDep = λ(name : Text) → mkDep name DepKind.Check

-- ============================================================================
-- Source Fetching
-- ============================================================================

let Src =
  < GitHub : { owner : Text, repo : Text, rev : Text, hash : SriHash }
  | GitLab : { owner : Text, repo : Text, rev : Text, hash : SriHash }
  | Url : { url : Text, hash : SriHash }
  | Git : { url : Text, rev : Text, hash : SriHash }
  | Store : StorePath
  | None
  >

-- ============================================================================
-- Outputs
-- ============================================================================

let Output =
  { name : Text
  , method : OutputMethod
  }

let floatingOut : Text → Output =
  λ(name : Text) → { name, method = OutputMethod.Floating }

let fixedOut : Text → SriHash → Output =
  λ(name : Text) → λ(hash : SriHash) → { name, method = OutputMethod.Fixed hash }

-- ============================================================================
-- Build Phases (typed, not shell)
-- ============================================================================

let Phases =
  { unpack : List Action
  , patch : List Action
  , configure : List Action
  , build : List Action
  , check : List Action
  , install : List Action
  , fixup : List Action
  }

let emptyPhases : Phases =
  { unpack = [] : List Action
  , patch = [] : List Action
  , configure = [] : List Action
  , build = [] : List Action
  , check = [] : List Action
  , install = [] : List Action
  , fixup = [] : List Action
  }

-- ============================================================================
-- The Derivation
-- ============================================================================

let Meta =
  { description : Text
  , homepage : Optional Text
  , license : Text
  , maintainers : List Text
  , platforms : List Text
  }

let Drv =
  { pname : Text
  , version : Text
  , system : Text
  
  -- CA-derivation specific
  , contentAddressed : Bool
  , outputs : List Output
  
  -- Source
  , src : Src
  
  -- Dependencies (structured, not [Text])
  , deps : List Dep
  
  -- Build phases (typed actions, not shell)
  , phases : Phases
  
  -- Environment variables
  , env : List { key : Text, value : Text }
  
  -- Metadata
  , meta : Meta
  
  -- Escape hatches
  , passthru : List { key : Text, value : Text }
  , shellHooks : { preBuild : Optional Text
                 , postBuild : Optional Text
                 , preInstall : Optional Text
                 , postInstall : Optional Text
                 }
  }

let defaultDrv : Drv =
  { pname = "unnamed"
  , version = "0.0.0"
  , system = "x86_64-linux"
  , contentAddressed = True
  , outputs = [floatingOut "out"]
  , src = Src.None
  , deps = [] : List Dep
  , phases = emptyPhases
  , env = [] : List { key : Text, value : Text }
  , meta = { description = ""
           , homepage = None Text
           , license = "unfree"
           , maintainers = [] : List Text
           , platforms = [] : List Text
           }
  , passthru = [] : List { key : Text, value : Text }
  , shellHooks = { preBuild = None Text
                 , postBuild = None Text
                 , preInstall = None Text
                 , postInstall = None Text
                 }
  }

-- ============================================================================
-- Action Helpers
-- ============================================================================

let copy = λ(s : Ref) → λ(d : Ref) → Action.Copy { src = s, dst = d }
let mkdir = λ(p : Ref) → Action.Mkdir { path = p, parents = True }
let symlink = λ(t : Ref) → λ(l : Ref) → Action.Symlink { target = t, link = l }
let write = λ(p : Ref) → λ(c : Text) → Action.Write { path = p, contents = c }

let installBin = λ(s : Ref) → Action.InstallBin { src = s }
let installLib = λ(s : Ref) → Action.InstallLib { src = s }
let installInclude = λ(s : Ref) → Action.InstallInclude { src = s }

let patchElfRpath = λ(p : Ref) → λ(rs : List Ref) → 
  Action.PatchElfRpath { path = p, rpaths = rs }

let substitute = λ(f : Ref) → λ(rs : List { from : Text, to : Text }) → 
  Action.Substitute { file = f, replacements = rs }

let run = λ(cmd : Ref) → λ(args : List Expr) →
  Action.Run 
    { cmd
    , args
    , env = [] : List { key : Text, value : Expr }
    , cwd = None Ref
    , stdin = None Ref
    , stdout = StreamTarget.Stdout
    , stderr = StreamTarget.Stderr
    }

let tool = λ(d : Text) → λ(b : Text) → λ(args : List Expr) →
  Action.Tool { dep = d, bin = b, args }

let cmake = λ(flags : List Text) →
  Action.CMake 
    { srcDir = src
    , buildDir = rel "build"
    , installPrefix = out
    , buildType = "Release"
    , flags
    , generator = < Ninja | Make | Default >.Ninja
    }

let make = λ(targets : List Text) →
  Action.Make 
    { targets
    , flags = [] : List Text
    , jobs = None Natural
    , dir = None Ref
    }

let shell = Action.Shell

-- ============================================================================
-- Exports
-- ============================================================================

in  { -- Types
      HashAlgo
    , Hash
    , SriHash
    , StorePath
    , mkStorePath
    , DrvPath
    , OutputMethod
    , Ref
    , Mode
    , Cmp
    , StreamTarget
    , Expr
    , Action
    , DepKind
    , Dep
    , Src
    , Output
    , Phases
    , Meta
    , Drv
    
    -- Ref constructors
    , dep
    , depSub
    , out
    , outSub
    , outNamed
    , src
    , srcSub
    , env
    , rel
    , lit
    , cat
    
    -- Dep constructors
    , mkDep
    , buildDep
    , hostDep
    , checkDep
    
    -- Output constructors
    , floatingOut
    , fixedOut
    
    -- Phases
    , emptyPhases
    
    -- Drv
    , defaultDrv
    
    -- Action helpers
    , copy
    , mkdir
    , symlink
    , write
    , installBin
    , installLib
    , installInclude
    , patchElfRpath
    , substitute
    , run
    , tool
    , cmake
    , make
    , shell
    }
