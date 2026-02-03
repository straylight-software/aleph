let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "no-translate-attrs-outside-prelude"
    , language = "nix"
    , severity = Severity.Error
    , rule = Schema.Rule::{
      , kind = "apply_expression"
      , any = Some
          [ Schema.SubRule::{
            , field = Some "function"
            , kind = Some "variable_expression"
            , has = Some
                ( nodeMatcher
                    { kind = Some "identifier"
                    , field = None Text
                    , regex = Some "^translate-attrs$"
                    , has = None Schema.NodeMatcher
                    , inside = None Schema.NodeMatcher
                    }
                )
            }
          ]
      }
    , message = "ALEPH-E013: translate-attrs used outside prelude"
    , note =
        ''
        ## What's wrong?
        `translate-attrs` is the prelude's internal translation layer.
        Using it directly means you're bypassing the typed interface.

        ## What can I do to fix this?
        Use `aleph.stdenv.default` or `prelude.mk-package` instead.
        ''
    , tests =
        { valid = [ "aleph.stdenv.default { }" ]
        , invalid = [ "translate-attrs { }" ]
        }
    }
