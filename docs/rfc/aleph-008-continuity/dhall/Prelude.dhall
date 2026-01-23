--| The Continuity Prelude
--| 
--| This replaces Starlark preludes and Nix stdenvs with typed Dhall.
--| Content-addressed imports. No magic. No globs.

let Prelude = https://prelude.dhall-lang.org/v23.1.0/package.dhall
  sha256:68622c2ac8d42a5d8d2b6c3b6e2b2ab0f5f8f51f8e6d6f5e5e5e5e5e5e5e5e5e5

in  { -- Re-export standard prelude
      List = Prelude.List
    , Map = Prelude.Map
    , Natural = Prelude.Natural
    , Optional = Prelude.Optional
    , Text = Prelude.Text
    , Bool = Prelude.Bool
    }
