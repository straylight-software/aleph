let Schema = ../../schemas/Lint.dhall

let node = Schema.someNodeMatcher

let Match = Schema.NodeMatcherWithDefaults

in  { id = "or-null-fallback"
    , language = "nix"
    , severity = Schema.Severity.Warning
    , rule = Schema.Rule::{
      , kind = "select_expression"
      , has =
          node
            Match::{
            , kind = Some "variable_expression"
            , field = Some "default"
            , has =
                node Match::{ kind = Some "identifier", regex = Some "^null\$" }
            }
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
      { valid =
        [ "{ foo ? null }: foo", "config.foo or true", "pkgs.bar or pkgs.baz" ]
      , invalid =
        [ "args.foo or null", "pkgs.hello or null", "config.value or null" ]
      }
    }
