/-
  Store.lean - Content-addressed store operations

  Uses Std.HashMap for efficient lookup.
  Demonstrates external library usage in multi-file project.
-/

import Straylight.Derivation
import Std.Data.HashMap

namespace Straylight

open Std (HashMap)

/-- The content-addressed store -/
structure Store where
  paths : HashMap Hash StorePath
  derivations : HashMap Hash Derivation
  deriving Inhabited

/-- Create an empty store -/
def Store.empty : Store :=
  { paths := {}, derivations := {} }

/-- Add a derivation to the store -/
def Store.addDerivation (s : Store) (d : Derivation) : Store :=
  let hash := d.hash
  let path := d.toStorePath
  { s with 
    paths := s.paths.insert hash path
    derivations := s.derivations.insert hash d }

/-- Look up a store path by hash -/
def Store.lookup (s : Store) (h : Hash) : Option StorePath :=
  s.paths[h]?

/-- Look up a derivation by hash -/
def Store.getDerivation (s : Store) (h : Hash) : Option Derivation :=
  s.derivations[h]?

/-- Check if a hash exists in the store -/
def Store.contains (s : Store) (h : Hash) : Bool :=
  s.paths.contains h

/-- Number of items in the store -/
def Store.size (s : Store) : Nat :=
  s.paths.size

/-- Build a derivation if not already in store -/
def Store.build (s : Store) (d : Derivation) : Store :=
  if s.contains d.hash then s
  else s.addDerivation d

/-- Get all store paths -/
def Store.allPaths (s : Store) : List StorePath :=
  s.paths.values

end Straylight
