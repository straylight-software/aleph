/-
  Proofs.lean - Theorems about derivations

  Proves key properties of content-addressed builds:
  - Determinism: same inputs -> same outputs
  - Referential transparency: derivations are pure functions

  Imports Derivation module.
-/

import Straylight.Derivation

namespace Straylight

/-- Two derivations with the same content have the same hash -/
theorem hash_deterministic (d1 d2 : Derivation) 
    (h_name : d1.name = d2.name)
    (h_inputs : d1.inputs = d2.inputs)
    (h_builder : d1.builder = d2.builder) :
    d1.hash = d2.hash := by
  simp [Derivation.hash, h_name, h_inputs, h_builder]

/-- A derivation's store path is determined by its content -/
theorem storePath_deterministic (d1 d2 : Derivation)
    (h : d1.hash = d2.hash) (h_name : d1.name = d2.name) :
    d1.toStorePath = d2.toStorePath := by
  simp [Derivation.toStorePath, h, h_name]

/-- The fundamental theorem: content-addressing ensures reproducibility -/
theorem content_addressed_reproducible 
    (d1 d2 : Derivation) 
    (h_eq : d1 = d2) : 
    d1.toStorePath = d2.toStorePath := by
  rw [h_eq]

/-- Empty inputs simplifies the hash computation -/
theorem empty_inputs_simplify (d : Derivation) (h : d.inputs = []) :
    d.hash = { value := d.name.length + d.builder.length } := by
  simp [Derivation.hash, h]

end Straylight
