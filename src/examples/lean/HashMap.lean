-- examples/lean/HashMap.lean
-- Test using Std.HashMap from Lean's standard library

import Std.Data.HashMap

def main : IO Unit := do
  let mut map : Std.HashMap String Nat := {}
  map := map.insert "straylight" 2026
  map := map.insert "aleph" 42
  IO.println s!"HashMap test: {map.size} entries"
  IO.println s!"straylight = {map["straylight"]?}"
