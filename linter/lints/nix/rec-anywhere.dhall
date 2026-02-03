let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "rec-anywhere"
    , language = "nix"
    , severity = Severity.Warning
    , rule = Schema.Rule::{ kind = "rec_attrset_expression" }
    , message = "ALEPH-W001: `rec` usage detected"
    , note =
        ''
        ## What's wrong?
        A recursive attrset (`rec { ... }`) was detected.

        ## What can I do to fix this?
        Consider using `let ... in` bindings or the fixed-point pattern instead.
        ''
    , tests =
      { valid = [ "let x = 1; in x", "{ a = 1; b = 2; }", "{ foo = bar; }" ]
      , invalid =
        [ "rec { a = 1; b = a; }"
        , "rec { x = 1; y = x + 1; }"
        , "rec { name = \"foo\"; path = ./. + name; }"
        ]
      }
    }
