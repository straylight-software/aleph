import Lake
open Lake DSL

package «continuity» where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩
  ]

-- Pin to mathlib version matching Lean 4.26.0
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.26.0"

@[default_target]
lean_lib «Continuity» where
  globs := #[.one `Continuity]

lean_lib «DhallEmbed» where
  globs := #[.one `DhallEmbed]
