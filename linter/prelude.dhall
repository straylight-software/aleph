{- Local minimal Prelude for dhall lint schema
   
   This avoids network dependencies during Nix builds.
   Only includes the functions we need from the standard Prelude.
-}

let List/map =
      λ(a : Type) → λ(b : Type) → λ(f : a → b) → λ(xs : List a) →
        List/build
          b
          ( λ(list : Type) → λ(cons : b → list → list) → λ(nil : list) →
              List/fold a xs list (λ(x : a) → cons (f x)) nil
          )

let List/filter =
      λ(a : Type) → λ(f : a → Bool) → λ(xs : List a) →
        List/build
          a
          ( λ(list : Type) → λ(cons : a → list → list) → λ(nil : list) →
              List/fold
                a
                xs
                list
                (λ(x : a) → λ(acc : list) → if f x then cons x acc else acc)
                nil
          )

let List/fold = List/fold

let List/build = List/build

let Optional/fold = Optional/fold

let Bool/not = λ(b : Bool) → if b then False else True

let Text/null = λ(t : Text) → Text/equal t ""

let Text/concatMapSep =
      λ(sep : Text) → λ(a : Type) → λ(f : a → Text) → λ(xs : List a) →
        List/fold a xs Text (λ(x : a) → λ(acc : Text) → acc ++ (if Text/null acc then "" else sep) ++ f x) ""

let Text/split = Text/split

let Map/Type = λ(a : Type) → List { mapKey : Text, mapValue : a }

in  { List = { map = List/map, filter = List/filter, fold = List/fold, build = List/build }
    , Optional = { fold = Optional/fold }
    , Bool = { not = Bool/not }
    , Text = { null = Text/null, concatMapSep = Text/concatMapSep, split = Text/split }
    , Map = { Type = Map/Type }
    , JSON = {-
        We need JSON from the standard Prelude. For now, we'll need to fetch it
        or provide a minimal implementation. Since JSON is complex, let's try a
        different approach - fetch the Prelude as a fixed-output derivation.
      -}
    }
