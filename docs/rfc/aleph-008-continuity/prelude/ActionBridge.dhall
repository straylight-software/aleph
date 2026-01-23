--| ActionBridge: Convert Script.dhall Commands to RFC-007 Actions
--|
--| This module bridges the gap between:
--| - RFC-008's Script.dhall (typed commands, Dhall)  
--| - RFC-007's Action type (Haskell, runs via builtins.wasm)
--|
--| The Action type is simpler (strings for paths) because it runs
--| inside the Nix sandbox where paths are already resolved.
--| The Script type is richer (typed paths) because it's the portable
--| representation that works across Nix, DICE, and standalone.
--|
--| This bridge enables:
--| 1. Write packages in Dhall with full type safety
--| 2. Compile to Haskell Action lists
--| 3. Run via existing builtins.wasm infrastructure

let Script = ./Script.dhall

-- =============================================================================
-- RFC-007 Action Type (must match Aleph.Nix.Derivation.Action exactly)
-- =============================================================================

-- | The Action type as defined in RFC-007 Aleph.Nix.Derivation
-- This is the "assembly language" that builtins.wasm understands
let Action =
      < WriteFile : { path : Text, content : Text }
      | Install : { mode : Natural, src : Text, dst : Text }
      | Mkdir : Text
      | Symlink : { target : Text, link : Text }
      | Copy : { src : Text, dst : Text }
      | Remove : Text
      | Unzip : Text
      | PatchElfRpath : { path : Text, rpaths : List Text }
      | PatchElfAddRpath : { path : Text, rpaths : List Text }
      | Substitute : { file : Text, replacements : List { from : Text, to : Text } }
      | Run : { cmd : Text, args : List Text }
      | ToolRun : { pkg : Text, args : List Text }
      >

-- =============================================================================
-- Path Resolution (Script.Path -> Text)
-- =============================================================================

-- | Resolve a Script.Path to a bash-compatible path string
let resolvePath : Script.Path -> Text =
      \(p : Script.Path) ->
        merge
          { Src = \(t : Text) -> if Text/null t then "\$src" else "\$src/${t}"
          , Out = \(t : Text) -> if Text/null t then "\$out" else "\$out/${t}"
          , Dep = \(d : { dep : Text, path : Text }) -> "\${${d.dep}}/${d.path}"
          , Tmp = \(t : Text) -> "\$TMPDIR/${t}"
          , Abs = \(t : Text) -> t
          }
          p

-- Helper for text equality check
let Text/null : Text -> Bool =
      \(t : Text) -> t == ""

-- =============================================================================
-- Interpolation Resolution (Script.Interp -> Text)
-- =============================================================================

let resolveInterp : Script.Interp -> Text =
      \(i : Script.Interp) ->
        merge
          { Lit = \(t : Text) -> t
          , Path = \(p : Script.Path) -> resolvePath p
          , Env = \(e : Script.Env) ->
              merge
                { Out = "\$out"
                , Src = "\$src"
                , Tmp = "\$TMPDIR"
                , Nix = \(v : Text) -> "\$NIX_${v}"
                , Var = \(v : Text) -> "\$${v}"
                }
                e
          , Dep = \(d : Text) -> "\${${d}}"
          , Join = \(j : { sep : Text, parts : List Script.Interp }) ->
              -- TODO: proper join, for now just concat
              ""
          }
          i

-- =============================================================================
-- Command -> Action Conversion
-- =============================================================================

-- | Convert a Script.Command to a list of Actions
-- Some commands map to multiple actions, hence List
let commandToActions : Script.Command -> List Action =
      \(cmd : Script.Command) ->
        merge
          { -- File operations
            Mkdir = \(m : { path : Script.Path, parents : Bool }) ->
              [ Action.Mkdir (resolvePath m.path) ]
          
          , Copy = \(c : { src : Script.Path, dst : Script.Path, recursive : Bool }) ->
              [ Action.Copy { src = resolvePath c.src, dst = resolvePath c.dst } ]
          
          , Move = \(_ : { src : Script.Path, dst : Script.Path }) ->
              -- Move = Copy + Remove (Action doesn't have Move)
              [] : List Action  -- TODO
          
          , Remove = \(r : { path : Script.Path, recursive : Bool, force : Bool }) ->
              [ Action.Remove (resolvePath r.path) ]
          
          , Symlink = \(s : { target : Script.Path, link : Script.Path }) ->
              [ Action.Symlink { target = resolvePath s.target, link = resolvePath s.link } ]
          
          , Chmod = \(_ : { path : Script.Path, mode : Script.Mode }) ->
              -- Chmod via Run (Action doesn't have Chmod)
              [] : List Action  -- TODO
          
          , Touch = \(t : { path : Script.Path }) ->
              [ Action.Run { cmd = "touch", args = [ resolvePath t.path ] } ]
          
          -- Content operations
          , Write = \(w : { path : Script.Path, content : Script.Interp }) ->
              [ Action.WriteFile { path = resolvePath w.path, content = resolveInterp w.content } ]
          
          , Append = \(_ : { path : Script.Path, content : Script.Interp }) ->
              -- Append via Run
              [] : List Action  -- TODO
          
          , Substitute = \(s : { file : Script.Path, replacements : List { from : Text, to : Script.Interp } }) ->
              let reps = List/map
                    { from : Text, to : Script.Interp }
                    { from : Text, to : Text }
                    (\(r : { from : Text, to : Script.Interp }) ->
                      { from = r.from, to = resolveInterp r.to })
                    s.replacements
              in [ Action.Substitute { file = resolvePath s.file, replacements = reps } ]
          
          -- Archive operations
          , Untar = \(u : { archive : Script.Path, dest : Script.Path, strip : Natural }) ->
              [ Action.Run
                  { cmd = "tar"
                  , args = [ "xf", resolvePath u.archive, "-C", resolvePath u.dest ]
                  }
              ]
          
          , Unzip = \(u : { archive : Script.Path, dest : Script.Path }) ->
              [ Action.Unzip (resolvePath u.dest) ]
          
          , Tar = \(_ : { files : List Script.Path, archive : Script.Path, compression : < None | Gzip | Xz | Zstd > }) ->
              [] : List Action  -- TODO
          
          -- Patching
          , Patch = \(p : { patch : Script.Path, strip : Natural }) ->
              [ Action.Run
                  { cmd = "patch"
                  , args = [ "-p${Natural/show p.strip}", "-i", resolvePath p.patch ]
                  }
              ]
          
          , PatchElf = \(pe : { binary : Script.Path, action : < SetRpath : List Script.Path | AddRpath : List Script.Path | SetInterpreter : Script.Path > }) ->
              merge
                { SetRpath = \(rpaths : List Script.Path) ->
                    [ Action.PatchElfRpath
                        { path = resolvePath pe.binary
                        , rpaths = List/map Script.Path Text resolvePath rpaths
                        }
                    ]
                , AddRpath = \(rpaths : List Script.Path) ->
                    [ Action.PatchElfAddRpath
                        { path = resolvePath pe.binary
                        , rpaths = List/map Script.Path Text resolvePath rpaths
                        }
                    ]
                , SetInterpreter = \(interp : Script.Path) ->
                    [ Action.Run
                        { cmd = "patchelf"
                        , args = [ "--set-interpreter", resolvePath interp, resolvePath pe.binary ]
                        }
                    ]
                }
                pe.action
          
          -- Build tools -> Run with appropriate command
          , Configure = \(c : { flags : List Script.Interp, workdir : Optional Script.Path }) ->
              [ Action.Run
                  { cmd = "./configure"
                  , args = List/map Script.Interp Text resolveInterp c.flags
                  }
              ]
          
          , Make = \(m : { targets : List Text, flags : List Script.Interp, jobs : Optional Natural }) ->
              [ Action.Run
                  { cmd = "make"
                  , args = m.targets # List/map Script.Interp Text resolveInterp m.flags
                  }
              ]
          
          , CMake = \(c : { srcdir : Script.Path, builddir : Script.Path, flags : List Script.Interp }) ->
              [ Action.Run
                  { cmd = "cmake"
                  , args = [ "-S", resolvePath c.srcdir, "-B", resolvePath c.builddir ]
                         # List/map Script.Interp Text resolveInterp c.flags
                  }
              ]
          
          , Meson = \(m : { srcdir : Script.Path, builddir : Script.Path, flags : List Script.Interp }) ->
              [ Action.Run
                  { cmd = "meson"
                  , args = [ "setup", resolvePath m.builddir, resolvePath m.srcdir ]
                         # List/map Script.Interp Text resolveInterp m.flags
                  }
              ]
          
          , Cargo = \(c : { command : < Build | Test | Install >, flags : List Script.Interp }) ->
              let subcmd = merge
                    { Build = "build"
                    , Test = "test"
                    , Install = "install"
                    }
                    c.command
              in [ Action.Run
                     { cmd = "cargo"
                     , args = [ subcmd ] # List/map Script.Interp Text resolveInterp c.flags
                     }
                 ]
          
          , Cabal = \(c : { command : < Build | Test | Install >, flags : List Script.Interp }) ->
              let subcmd = merge
                    { Build = "build"
                    , Test = "test"
                    , Install = "install"
                    }
                    c.command
              in [ Action.Run
                     { cmd = "cabal"
                     , args = [ subcmd ] # List/map Script.Interp Text resolveInterp c.flags
                     }
                 ]
          
          -- Install helpers -> Install action
          , InstallBin = \(i : { src : Script.Path, name : Optional Text }) ->
              [ Action.Install
                  { mode = 0o755  -- 493 in decimal
                  , src = resolvePath i.src
                  , dst = merge { Some = \(n : Text) -> "bin/${n}", None = "bin/" } i.name
                  }
              ]
          
          , InstallLib = \(i : { src : Script.Path, name : Optional Text }) ->
              [ Action.Install
                  { mode = 0o644  -- 420 in decimal
                  , src = resolvePath i.src
                  , dst = merge { Some = \(n : Text) -> "lib/${n}", None = "lib/" } i.name
                  }
              ]
          
          , InstallHeader = \(i : { src : Script.Path, name : Optional Text }) ->
              [ Action.Install
                  { mode = 0o644
                  , src = resolvePath i.src
                  , dst = merge { Some = \(n : Text) -> "include/${n}", None = "include/" } i.name
                  }
              ]
          
          , InstallMan = \(i : { src : Script.Path, section : Natural }) ->
              [ Action.Install
                  { mode = 0o644
                  , src = resolvePath i.src
                  , dst = "share/man/man${Natural/show i.section}/"
                  }
              ]
          
          , InstallDoc = \(i : { src : Script.Path }) ->
              [ Action.Install
                  { mode = 0o644
                  , src = resolvePath i.src
                  , dst = "share/doc/"
                  }
              ]
          
          -- Control flow -> flattened
          , If = \(_ : { cond : Script.Condition, then_ : List Script.Command, else_ : List Script.Command }) ->
              -- Control flow requires runtime, emit as Run
              [] : List Action  -- TODO: could emit shell
          
          , For = \(_ : { var : Text, in_ : List Script.Interp, do_ : List Script.Command }) ->
              [] : List Action  -- TODO
          
          -- Escape hatches
          , Run = \(r : { cmd : Text, args : List Script.Interp, env : List { name : Text, value : Script.Interp } }) ->
              [ Action.Run
                  { cmd = r.cmd
                  , args = List/map Script.Interp Text resolveInterp r.args
                  }
              ]
          
          , Shell = \(script : Text) ->
              -- Shell escape hatch - emit as Run with sh -c
              [ Action.Run { cmd = "sh", args = [ "-c", script ] } ]
          }
          cmd

-- | Convert a Script to a list of Actions (flattening)
let scriptToActions : Script.Script -> List Action =
      \(script : Script.Script) ->
        List/fold
          Script.Command
          script
          (List Action)
          (\(cmd : Script.Command) -> \(acc : List Action) ->
            acc # commandToActions cmd)
          ([] : List Action)

-- =============================================================================
-- Exports
-- =============================================================================

in  { Action
    , resolvePath
    , resolveInterp
    , commandToActions
    , scriptToActions
    }
