let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

let node = Schema.someNodeMatcher

let Match = Schema.NodeMatcherWithDefaults

in  { id = "no-substitute-all"
    , language = "nix"
    , severity = Severity.Error
    , rule = Schema.Rule::{
      , kind = "apply_expression"
      , any = Some
        [ Schema.SubRule::{
          , field = Some "function"
          , kind = Some "select_expression"
          , has =
              node
                Match::{
                , kind = Some "attrpath"
                , field = Some "attrpath"
                , has =
                    node
                      Match::{
                      , kind = Some "identifier"
                      , regex = Some "^(substituteAll|replaceVars|substitute)\$"
                      }
                }
          }
        , Schema.SubRule::{
          , field = Some "function"
          , kind = Some "variable_expression"
          , has =
              node
                Match::{
                , kind = Some "identifier"
                , regex = Some "^(substituteAll|replaceVars|substitute)\$"
                }
          }
        ]
      }
    , message = "ALEPH-E007: Text templating must use Dhall"
    , note =
        ''
        ## What's wrong?
        `substituteAll`, `replaceVars`, and `substitute` are forbidden.
        All text generation must use Dhall templates.
        ''
    , tests =
      { valid =
        [ "builtins.readFile ./file"
        , "\"plain text\""
        , "pkgs.writeText \"config\" configText"
        ]
      , invalid =
        [ "substituteAll { src = ./file; }"
        , "replaceVars ./template { var = value; }"
        , "substitute { src = script.sh; }"
        ]
      }
    }
