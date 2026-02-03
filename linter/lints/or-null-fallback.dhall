let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "or-null-fallback"
    , language = "nix"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "select_expression"
      , has = Some
          ( nodeMatcher
              { kind = Some "variable_expression"
              , field = Some "default"
              , regex = None Text
              , has = Some
                  ( nodeMatcher
                      { kind = Some "identifier"
                      , field = None Text
                      , regex = Some "^null$"
                      , has = None Schema.NodeMatcher
                      , inside = None Schema.NodeMatcher
                      }
                  )
              , inside = None Schema.NodeMatcher
              }
          )
      }
    , message = "ALEPH-W004: defensive `or null` fallback"
    , note =
        ''
        ## What's wrong?
        Using `x or null` as a fallback hides errors instead of failing fast.

        ## What can I do to fix this?
        If the attribute must exist, remove the fallback and let it fail.
        ''
    , tests =
        { valid = [ "{ foo ? null }: foo" ]
        , invalid = [ "args.foo or null" ]
        }
    }
