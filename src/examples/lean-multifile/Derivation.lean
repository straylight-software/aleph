/-
  Derivation.lean - Content-addressed derivation model

  Defines the core types for Straylight's build system:
  - Hash: content addresses
  - Derivation: build units with inputs/outputs
  - Store: the content-addressed store

  This module has no dependencies beyond Init.
-/

namespace Straylight

/-- A content hash (simplified as Nat for this example) -/
structure Hash where
  value : Nat
  deriving Repr, DecidableEq, Hashable

/-- Create a hash from a string (simplified) -/
def Hash.ofString (s : String) : Hash :=
  { value := s.foldl (fun acc c => acc * 31 + c.toNat) 0 }

/-- A derivation input: either a source file or another derivation's output -/
inductive Input where
  | source : String -> Hash -> Input
  | dep : Hash -> Input
  deriving Repr

/-- A derivation represents a build step -/
structure Derivation where
  name : String
  inputs : List Input
  builder : String
  args : List String
  outputHash : Hash
  deriving Repr

/-- Compute the content hash of a derivation (its "drv hash") -/
def Derivation.hash (d : Derivation) : Hash :=
  let inputHashes := d.inputs.map fun
    | .source _ h => h.value
    | .dep h => h.value
  let combined := inputHashes.foldl (· + ·) 0
  { value := combined + d.name.length + d.builder.length }

/-- A store path in the content-addressed store -/
structure StorePath where
  hash : Hash
  name : String
  deriving Repr

/-- Convert a derivation to its store path -/
def Derivation.toStorePath (d : Derivation) : StorePath :=
  { hash := d.hash, name := d.name }

end Straylight
