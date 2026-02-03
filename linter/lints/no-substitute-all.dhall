let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "no-substitute-all"
    , language = "nix"
    , severity = Severity.Error
    , rule = Schema.Rule::{
      , kind = "apply_expression"
      , any = Some
          [ Schema.SubRule::{
            , field = Some "function"
            , kind = Some "select_expression"
            , has = Some
                ( nodeMatcher
                    { kind = Some "attrpath"
                    , field = Some "attrpath"
                    , regex = None Text
                    , has = Some
                        ( nodeMatcher
                            { kind = Some "identifier"
                            , field = None Text
                            , regex = Some "^(substituteAll|replaceVars|substitute)$"
                            , has = None Schema.NodeMatcher
                            , inside = None Schema.NodeMatcher
                            }
                        )
                    , inside = None Schema.NodeMatcher
                    }
                )
            }
          , Schema.SubRule::{
            , field = Some "function"
            , kind = Some "variable_expression"
            , has = Some
                ( nodeMatcher
                    { kind = Some "identifier"
                    , field = None Text
                    , regex = Some "^(substituteAll|replaceVars|substitute)$"
                    , has = None Schema.NodeMatcher
                    , inside = None Schema.NodeMatcher
                    }
                )
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
        { valid = [ "builtins.readFile ./file" ]
        , invalid = [ "substituteAll { src = ./file; }" ]
        }
    }
