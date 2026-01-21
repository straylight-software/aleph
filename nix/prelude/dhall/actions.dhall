-- nix/prelude/dhall/actions.dhall
--
-- Typed build actions for derivation phases.
--
-- These are NOT converted to shell strings. They are serialized as Dhall,
-- validated by Nix (store paths checked), then executed directly by aleph-exec.
--
-- There is no "Run" escape hatch. If you need a new action, add it here.

let StorePath = ./store-path.dhall

-- Replacement pair for Substitute action
let Replacement : Type =
  { from : Text
  , to : Text
  }

-- Wrapper actions for the Wrap action
let WrapAction =
  < Prefix : { var : Text, value : Text }
  | Suffix : { var : Text, value : Text }
  | Set : { var : Text, value : Text }
  | SetDefault : { var : Text, value : Text }
  | Unset : { var : Text }
  | AddFlags : { flags : Text }
  >

-- The core Action type - all build operations
let Action =
  -- Create a directory (mkdir -p $out/<path>)
  < Mkdir : { path : Text }

  -- Copy file or directory
  | Copy : { from : Text, to : Text }

  -- Create symbolic link
  | Symlink : { target : Text, link : Text }

  -- Write file with content
  | WriteFile : { path : Text, content : Text }

  -- Install file with mode (replaces install -m)
  | Install : { mode : Natural, src : Text, dst : Text }

  -- Remove file or directory
  | Remove : { path : Text }

  -- Extract zip archive ($src) to destination
  | Unzip : { dest : Text }

  -- Set ELF rpath (paths are StorePaths, validated)
  | PatchElfRpath : { binary : Text, rpaths : List StorePath.StorePath }

  -- Add to ELF rpath
  | PatchElfAddRpath : { binary : Text, rpaths : List StorePath.StorePath }

  -- Set ELF interpreter
  | PatchElfInterpreter : { binary : Text, interpreter : StorePath.StorePath }

  -- Substitute strings in file
  | Substitute : { file : Text, replacements : List Replacement }

  -- Wrap program with environment modifications
  | Wrap : { program : Text, actions : List WrapAction }

  -- Chmod (when Install mode isn't sufficient)
  | Chmod : { path : Text, mode : Natural }
  >

-- A phase is a list of actions executed in order
let Phase : Type = List Action

-- Convenience constructors
let mkdir : Text -> Action =
  \(path : Text) -> Action.Mkdir { path }

let copy : Text -> Text -> Action =
  \(from : Text) -> \(to : Text) -> Action.Copy { from, to }

let symlink : Text -> Text -> Action =
  \(target : Text) -> \(link : Text) -> Action.Symlink { target, link }

let writeFile : Text -> Text -> Action =
  \(path : Text) -> \(content : Text) -> Action.WriteFile { path, content }

let install : Natural -> Text -> Text -> Action =
  \(mode : Natural) -> \(src : Text) -> \(dst : Text) ->
    Action.Install { mode, src, dst }

let installBin : Text -> Text -> Action = install 0o755
let installLib : Text -> Text -> Action = install 0o644
let installHeader : Text -> Text -> Action = install 0o644
let installData : Text -> Text -> Action = install 0o644

let remove : Text -> Action =
  \(path : Text) -> Action.Remove { path }

let unzip : Text -> Action =
  \(dest : Text) -> Action.Unzip { dest }

let patchElfRpath : Text -> List StorePath.StorePath -> Action =
  \(binary : Text) -> \(rpaths : List StorePath.StorePath) ->
    Action.PatchElfRpath { binary, rpaths }

let substitute : Text -> List Replacement -> Action =
  \(file : Text) -> \(replacements : List Replacement) ->
    Action.Substitute { file, replacements }

let replace : Text -> Text -> Replacement =
  \(from : Text) -> \(to : Text) -> { from, to }

let wrap : Text -> List WrapAction -> Action =
  \(program : Text) -> \(actions : List WrapAction) ->
    Action.Wrap { program, actions }

let wrapPrefix : Text -> Text -> WrapAction =
  \(var : Text) -> \(value : Text) -> WrapAction.Prefix { var, value }

let wrapSet : Text -> Text -> WrapAction =
  \(var : Text) -> \(value : Text) -> WrapAction.Set { var, value }

let chmod : Text -> Natural -> Action =
  \(path : Text) -> \(mode : Natural) -> Action.Chmod { path, mode }

in
{ StorePath = StorePath.StorePath
, Replacement
, WrapAction
, Action
, Phase
-- Constructors
, mkdir
, copy
, symlink
, writeFile
, install
, installBin
, installLib
, installHeader
, installData
, remove
, unzip
, patchElfRpath
, substitute
, replace
, wrap
, wrapPrefix
, wrapSet
, chmod
}
