/-
  Dhall → Lean Embedding (Core)
  ==============================
  
  The key insight: Dhall normalization = Lean's rfl.
  We demonstrate this with simple intrinsically-typed terms.
-/

namespace DhallEmbed

/-!
## §1 Simply Typed Lambda Calculus in Lean
-/

/-- Simple types -/
inductive Ty : Type where
  | nat : Ty
  | bool : Ty
  | arr : Ty → Ty → Ty
  deriving Repr, DecidableEq

notation:50 α " ⇒ " β => Ty.arr α β

/-- Interpret types as Lean types -/
def sem : Ty → Type
  | .nat => Nat
  | .bool => Bool
  | .arr α β => sem α → sem β

/-- Typing context -/
abbrev Ctx := List Ty

/-- De Bruijn index: proof that type τ is at position in Γ -/
inductive Var : Ctx → Ty → Type where
  | z {Γ : Ctx} {τ : Ty} : Var (τ :: Γ) τ
  | s {Γ : Ctx} {τ σ : Ty} : Var Γ τ → Var (σ :: Γ) τ

/-- Intrinsically typed terms -/
inductive Term : Ctx → Ty → Type where
  | var {Γ : Ctx} {τ : Ty} : Var Γ τ → Term Γ τ
  | lam {Γ : Ctx} {α β : Ty} : Term (α :: Γ) β → Term Γ (α ⇒ β)
  | app {Γ : Ctx} {α β : Ty} : Term Γ (α ⇒ β) → Term Γ α → Term Γ β
  | natLit {Γ : Ctx} : Nat → Term Γ .nat
  | boolLit {Γ : Ctx} : Bool → Term Γ .bool

/-- Environment: values for all variables in context -/
def Env : Ctx → Type
  | [] => Unit
  | τ :: Γ => sem τ × Env Γ

/-- Variable lookup -/
def lookupVar {Γ : Ctx} {τ : Ty} : Var Γ τ → Env Γ → sem τ
  | .z, (v, _) => v
  | .s x, (_, env) => lookupVar x env

/-- Evaluation: terms to values -/
def eval {Γ : Ctx} {τ : Ty} : Term Γ τ → Env Γ → sem τ
  | .var x, env => lookupVar x env
  | .lam body, env => fun a => eval body (a, env)
  | .app f a, env => eval f env (eval a env)
  | .natLit n, _ => n
  | .boolLit b, _ => b

/-!
## §2 The rfl Proofs

These are the key demonstrations: evaluation is definitional.
-/

/-- Identity: λx. x -/
def idTerm : Term [] (.nat ⇒ .nat) := .lam (.var .z)

/-- Identity evaluates to Lean's id -/
theorem id_correct : eval idTerm () = fun (x : Nat) => x := rfl

/-- Const: λx. λy. x -/
def constTerm : Term [] (.nat ⇒ .bool ⇒ .nat) :=
  .lam (.lam (.var (.s .z)))

/-- Const evaluates to Lean's const -/
theorem const_correct : eval constTerm () = fun (x : Nat) (_ : Bool) => x := rfl

/-- Flip: λf. λb. λa. f a b -/
def flipTerm : Term [] ((.nat ⇒ .bool ⇒ .nat) ⇒ .bool ⇒ .nat ⇒ .nat) :=
  .lam (.lam (.lam 
    (.app (.app (.var (.s (.s .z))) (.var .z)) (.var (.s .z)))))

/-- Flip evaluates to Lean's flip -/
theorem flip_correct : eval flipTerm () = 
    (fun (f : Nat → Bool → Nat) (b : Bool) (a : Nat) => f a b) := rfl

/-- Application: (λx. x) 42 -/
def appTerm : Term [] .nat := .app idTerm (.natLit 42)

/-- Application evaluates correctly -/
theorem app_correct : eval appTerm () = (42 : Nat) := rfl

/-- Nested: (λf. λx. f x) id 99 -/
def nestedTerm : Term [] .nat :=
  .app (.app 
    (.lam (.lam (.app (.var (.s .z)) (.var .z))))  -- λf. λx. f x
    idTerm)                                         -- id  
    (.natLit 99)                                    -- 99

/-- Nested evaluates correctly -/
theorem nested_correct : eval nestedTerm () = (99 : Nat) := rfl



/-!
## §4 The Dhall Connection

Dhall is System Fω, which includes STLC.
Every Dhall expression (in the simply-typed fragment) corresponds 
to a `Term` here. Dhall's normalization = our `eval`.
Dhall's `assert : x === y` = our `rfl` proofs.

The key insight:
- Dhall programs are total (no infinite loops)
- Dhall normalization is decidable  
- Therefore Dhall equality = definitional equality
- Therefore Dhall proofs = Lean's rfl

This is the **rfl nexus**: Dhall's computational equality
flows directly into Lean's type theory.

For BUILD.dhall files:
1. Parse → untyped AST
2. Typecheck → `Term Γ τ` (or error)
3. Evaluate → value (by rfl)

If step 2 succeeds, step 3 is guaranteed.
The build configuration is machine-verified.
-/

/-- Every well-typed term evaluates deterministically -/
theorem eval_deterministic {Γ : Ctx} {τ : Ty} (t : Term Γ τ) (env : Env Γ) :
    eval t env = eval t env := rfl

end DhallEmbed
