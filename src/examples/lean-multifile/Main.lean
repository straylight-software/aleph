/-
  Main.lean - Straylight derivation demo

  Multi-file Lean 4 program demonstrating:
  - Module imports (Derivation, Store, Proofs)
  - External library (Std.HashMap)
  - Proof-carrying code
  - Executable binary
-/

import Straylight.Derivation
import Straylight.Store
import Straylight.Proofs

open Straylight

def main : IO Unit := do
  IO.println "=============================================="
  IO.println "  Straylight Derivation Model - Lean 4.26.0"
  IO.println "=============================================="
  IO.println ""

  -- Create some derivations
  let srcHash := Hash.ofString "hello.c"
  let src := Input.source "hello.c" srcHash
  
  let d1 : Derivation := {
    name := "hello"
    inputs := [src]
    builder := "gcc"
    args := ["-o", "hello", "hello.c"]
    outputHash := Hash.ofString "hello-output"
  }
  
  let d2 : Derivation := {
    name := "world"
    inputs := [.dep d1.hash]
    builder := "gcc"
    args := ["-o", "world", "world.c"]
    outputHash := Hash.ofString "world-output"
  }

  IO.println s!"Derivation 1: {d1.name}"
  IO.println s!"  Hash: {d1.hash.value}"
  IO.println s!"  Store path: {d1.toStorePath.name}"
  IO.println ""

  IO.println s!"Derivation 2: {d2.name}"
  IO.println s!"  Hash: {d2.hash.value}"
  IO.println s!"  Depends on: {d1.name}"
  IO.println ""

  -- Build a store
  IO.println "Building content-addressed store..."
  let store := Store.empty
    |>.addDerivation d1
    |>.addDerivation d2
  
  IO.println s!"Store size: {store.size} derivations"
  IO.println ""

  -- Demonstrate lookup
  IO.println "Looking up d1 by hash..."
  match store.lookup d1.hash with
  | some path => IO.println s!"  Found: {path.name}"
  | none => IO.println "  Not found"

  -- Verify theorems apply
  IO.println ""
  IO.println "Proof verification:"
  IO.println "  - hash_deterministic: verified at compile time"
  IO.println "  - storePath_deterministic: verified at compile time"
  IO.println "  - content_addressed_reproducible: verified at compile time"
  IO.println ""

  -- Show all paths
  IO.println "All store paths:"
  for path in store.allPaths do
    IO.println s!"  /nix/store/{path.hash.value}-{path.name}"

  IO.println ""
  IO.println "=============================================="
  IO.println "  Build system: hermetic, content-addressed"
  IO.println "=============================================="
