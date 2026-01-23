--| Typed Build Scripts
--|
--| The escape hatch that ruins everything is shell scripts.
--| This module defines a typed command language with clean interpolation.
--|
--| Key properties:
--| 1. No string interpolation bugs
--| 2. No quoting errors  
--| 3. Every command is auditable
--| 4. Compiles to bash OR builtins.wasm actions

let Prelude = https://prelude.dhall-lang.org/v23.1.0/package.dhall
    sha256:931cbfae9d746c4611b07633ab1e547637ab4ba138b16bf65ef1b9ad66a60b7f

let List/map = Prelude.List.map

let Types = ./Types.dhall

-- =============================================================================
-- Path Types (the key insight: paths are not strings)
-- =============================================================================

let Path =
      < Src : Text              -- Source tree relative path
      | Out : Text              -- $out relative path
      | Dep : { dep : Text, path : Text }  -- Dependency output path
      | Tmp : Text              -- $TMPDIR relative path
      | Abs : Text              -- Escape hatch: absolute path (logged)
      >

let Env =
      < Out                     -- $out
      | Src                     -- $src (if applicable)
      | Tmp                     -- $TMPDIR
      | Nix : Text              -- $NIX_* variable
      | Var : Text              -- Named variable (logged)
      >

-- =============================================================================
-- Interpolation (typed, not stringly)
-- Note: We avoid recursive types here for simplicity. Use JoinText for joining.
-- =============================================================================

let Interp =
      < Lit : Text              -- Literal text
      | Path : Path             -- A path (will be quoted correctly)
      | Env : Env               -- An environment variable
      | Dep : Text              -- A dependency reference (//path:name)
      >

let lit = \(t : Text) -> Interp.Lit t
let path = \(p : Path) -> Interp.Path p
let env = \(e : Env) -> Interp.Env e
let out = Interp.Path (Path.Out "")
let src = Interp.Env Env.Src
let tmp = Interp.Env Env.Tmp

-- =============================================================================
-- File Operations (typed, not bash)
-- =============================================================================

let FileMode = < Read | Write | Execute | All >

let Mode =
      { owner : List FileMode
      , group : List FileMode
      , other : List FileMode
      }

let mode755 : Mode =
      { owner = [ FileMode.Read, FileMode.Write, FileMode.Execute ]
      , group = [ FileMode.Read, FileMode.Execute ]
      , other = [ FileMode.Read, FileMode.Execute ]
      }

let mode644 : Mode =
      { owner = [ FileMode.Read, FileMode.Write ]
      , group = [ FileMode.Read ]
      , other = [ FileMode.Read ]
      }

-- =============================================================================
-- Conditions (for If)
-- =============================================================================

let Condition =
      < PathExists : Path
      | FileExists : Path
      | DirExists : Path
      | EnvSet : Text
      | EnvEquals : { var : Text, value : Text }
      | True
      | False
      >

-- =============================================================================
-- The Command Type (what you can actually do)
-- =============================================================================

let Command =
      -- File operations
      < Mkdir : { path : Path, parents : Bool }
      | Copy : { src : Path, dst : Path, recursive : Bool }
      | Move : { src : Path, dst : Path }
      | Remove : { path : Path, recursive : Bool, force : Bool }
      | Symlink : { target : Path, link : Path }
      | Chmod : { path : Path, mode : Mode }
      | Touch : { path : Path }
      
      -- Content operations
      | Write : { path : Path, content : Interp }
      | Append : { path : Path, content : Interp }
      | Substitute : { file : Path, replacements : List { from : Text, to : Interp } }
      
      -- Archive operations  
      | Untar : { archive : Path, dest : Path, strip : Natural }
      | Unzip : { archive : Path, dest : Path }
      | Tar : { files : List Path, archive : Path, compression : < None | Gzip | Xz | Zstd > }
      
      -- Patching
      | Patch : { patch : Path, strip : Natural }
      | PatchElf : { binary : Path, action : < SetRpath : List Path | AddRpath : List Path | SetInterpreter : Path > }
      
      -- Build tool wrappers (typed invocations)
      | Configure : { flags : List Interp, workdir : Optional Path }
      | Make : { targets : List Text, flags : List Interp, jobs : Optional Natural }
      | CMake : { srcdir : Path, builddir : Path, flags : List Interp }
      | Meson : { srcdir : Path, builddir : Path, flags : List Interp }
      | Cargo : { command : < Build | Test | Install >, flags : List Interp }
      | Cabal : { command : < Build | Test | Install >, flags : List Interp }
      
      -- Install helpers
      | InstallBin : { src : Path, name : Optional Text }
      | InstallLib : { src : Path, name : Optional Text }
      | InstallHeader : { src : Path, name : Optional Text }
      | InstallMan : { src : Path, section : Natural }
      | InstallDoc : { src : Path }
      
      -- Escape hatch (logged and warned)
      | Run : { cmd : Text, args : List Interp, env : List { name : Text, value : Interp } }
      | Shell : Text  -- Raw shell (LOUD warning)
      >

-- =============================================================================
-- Script (a sequence of commands)
-- =============================================================================

let Script = List Command

-- =============================================================================
-- Smart Constructors
-- =============================================================================

let mkdir =
      \(p : Text) ->
        Command.Mkdir { path = Path.Out p, parents = True }

let mkdirp = mkdir

let copy =
      \(src : Text) ->
      \(dst : Text) ->
        Command.Copy { src = Path.Src src, dst = Path.Out dst, recursive = True }

let copyFile =
      \(src : Text) ->
      \(dst : Text) ->
        Command.Copy { src = Path.Src src, dst = Path.Out dst, recursive = False }

let symlink =
      \(target : Text) ->
      \(link : Text) ->
        Command.Symlink { target = Path.Out target, link = Path.Out link }

let write =
      \(path : Text) ->
      \(content : Text) ->
        Command.Write { path = Path.Out path, content = Interp.Lit content }

let untar =
      \(archive : Text) ->
      \(dest : Text) ->
        Command.Untar { archive = Path.Src archive, dest = Path.Out dest, strip = 0 }

let unzip =
      \(dest : Text) ->
        Command.Unzip { archive = Path.Src "", dest = Path.Out dest }

let installBin =
      \(src : Text) ->
        Command.InstallBin { src = Path.Src src, name = None Text }

let installLib =
      \(src : Text) ->
        Command.InstallLib { src = Path.Src src, name = None Text }

let configure =
      \(flags : List Text) ->
        Command.Configure 
          { flags = List/map Text Interp Interp.Lit flags
          , workdir = None Path
          }

let make =
      \(targets : List Text) ->
        Command.Make { targets, flags = [] : List Interp, jobs = None Natural }

let makeInstall = make [ "install" ]

let cmake =
      \(flags : List Text) ->
        Command.CMake
          { srcdir = Path.Src "."
          , builddir = Path.Tmp "build"
          , flags = List/map Text Interp Interp.Lit flags
          }

let run =
      \(cmd : Text) ->
      \(args : List Text) ->
        Command.Run
          { cmd
          , args = List/map Text Interp Interp.Lit args
          , env = [] : List { name : Text, value : Interp }
          }

-- =============================================================================
-- Phase Builders
-- =============================================================================

let Phase =
      { name : Text
      , commands : Script
      }

let unpackPhase =
      \(commands : Script) ->
        { name = "unpack", commands } : Phase

let patchPhase =
      \(commands : Script) ->
        { name = "patch", commands } : Phase

let configurePhase =
      \(commands : Script) ->
        { name = "configure", commands } : Phase

let buildPhase =
      \(commands : Script) ->
        { name = "build", commands } : Phase

let checkPhase =
      \(commands : Script) ->
        { name = "check", commands } : Phase

let installPhase =
      \(commands : Script) ->
        { name = "install", commands } : Phase

let fixupPhase =
      \(commands : Script) ->
        { name = "fixup", commands } : Phase

-- =============================================================================
-- Exports
-- =============================================================================

in  { Path
    , Env
    , Interp
    , lit
    , path
    , env
    , out
    , src
    , tmp
    , FileMode
    , Mode
    , mode755
    , mode644
    , Command
    , Condition
    , Script
    -- Smart constructors
    , mkdir
    , mkdirp
    , copy
    , copyFile
    , symlink
    , write
    , untar
    , unzip
    , installBin
    , installLib
    , configure
    , make
    , makeInstall
    , cmake
    , run
    -- Phases
    , Phase
    , unpackPhase
    , patchPhase
    , configurePhase
    , buildPhase
    , checkPhase
    , installPhase
    , fixupPhase
    }
