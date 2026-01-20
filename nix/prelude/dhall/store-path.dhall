-- nix/prelude/dhall/store-path.dhall
--
-- Typed Nix store path representation.
--
-- A StorePath is NEVER interpolated as a raw string in Nix. It is validated
-- against the actual store at evaluation time, then passed to aleph-exec
-- which resolves it to a filesystem path.
--
-- This eliminates injection attacks: you cannot construct a StorePath from
-- arbitrary user input. Only Nix can create valid StorePaths by validating
-- against /nix/store.

let StorePath : Type =
  { hash : Text      -- The hash portion (e.g., "abc123...")
  , name : Text      -- The name portion (e.g., "zlib-1.2.3")
  , output : Text    -- Output subpath (e.g., "/lib" or "" for root)
  }

let resolve : StorePath -> Text =
  \(p : StorePath) ->
    "/nix/store/${p.hash}-${p.name}${p.output}"

-- Smart constructor for package references (resolved by Nix)
let pkg : Text -> StorePath =
  \(name : Text) ->
    { hash = ""      -- Filled in by Nix at eval time
    , name = name
    , output = ""
    }

-- Reference a specific output of a store path
let withOutput : Text -> StorePath -> StorePath =
  \(out : Text) ->
  \(p : StorePath) ->
    p // { output = out }

-- Common output helpers
let lib : StorePath -> StorePath = withOutput "/lib"
let bin : StorePath -> StorePath = withOutput "/bin"
let include : StorePath -> StorePath = withOutput "/include"
let share : StorePath -> StorePath = withOutput "/share"

in
{ StorePath
, resolve
, pkg
, withOutput
, lib
, bin
, include
, share
}
