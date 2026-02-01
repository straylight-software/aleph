-- Ast-grep rule schema for Dhall (simplified - no recursive types)
-- This provides types for all ast-grep rule constructs

let RuleKind : Type = Text

let Severity : Type = Text

let Position : Type = Text

let Pattern : Type = Text

let Regex : Type = Text

-- For recursive structures, we use a simpler approach
-- Just define the fields we commonly use
let BaseRelation : Type =
      { kind : Optional RuleKind
      , pattern : Optional Pattern
      , regex : Optional Regex
      , field : Optional Position
      }

let defaultBaseRelation : BaseRelation =
      { kind = None RuleKind
      , pattern = None Pattern
      , regex = None Regex
      , field = None Position
      }

let Rule : Type =
      { id : Text
      , language : Text
      , severity : Optional Severity
      , rule : BaseRelation
      , message : Text
      , note : Optional Text
      }

let TestCase : Type =
      { id : Text
      , valid : Optional (List Text)
      , invalid : Optional (List Text)
      }

in  { RuleKind
    , Severity
    , Position
    , Pattern
    , Regex
    , BaseRelation
    , defaultBaseRelation
    , Rule
    , TestCase
    }
