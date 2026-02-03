let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "rec-anywhere"
    , language = "nix"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "rec_attrset_expression"
        , regex = None Text
        , pattern = None Text
        , has = None Schema.NodeMatcher
        , not = None Schema.RuleNot
        }
    , message = "ALEPH-W001: `rec` usage detected"
    , note =
        ''
        ## What's wrong?
        A recursive attrset (`rec { ... }`) was detected.

        This is discouraged because it:
        - Makes it harder to reason about evaluation order
        - Can lead to infinite recursion bugs

        ## What can I do to fix this?
        Consider using `let ... in` bindings or the fixed-point pattern instead.
        ''
    , tests =
        { valid =
            [ "let x = 1; in x"
            , "{ a = 1; b = 2; }"
            ]
        , invalid =
            [ "rec { a = 1; b = a; }"
            ]
        }
    }
