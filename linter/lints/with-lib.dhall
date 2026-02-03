let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "with-lib"
    , language = "nix"
    , severity = Severity.Error
    , rule =
        { kind = "with_expression"
        , regex = None Text
        , pattern = None Text
        , has = Some
            ( nodeMatcher
                { kind = Some "variable_expression"
                , field = Some "environment"
                , regex = None Text
                , has = Some
                    ( nodeMatcher
                        { kind = Some "identifier"
                        , field = None Text
                        , regex = Some "^lib$"
                        , has = None Schema.NodeMatcher
                        , inside = None Schema.NodeMatcher
                        }
                    )
                , inside = None Schema.NodeMatcher
                }
            )
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-E001: `with lib;` statement"
    , note =
        ''
        ## What's wrong?
        Usage of the `with lib;` construct was detected.

        This is forbidden because it:
        - Obscures where names come from
        - Breaks tooling (no go-to-definition, no accurate autocomplete)
        - Creates shadowing hazards when lib adds new attributes
        - Makes code review require mental scope tracking

        ## What can I do to fix this?
        Try something like this instead:

        ```nix
        inherit (lib) types mkOption;
        ```
        ''
    , tests =
        { valid =
            [ "environment.systemPackages = with pkgs; [ vim git ];"
            ]
        , invalid =
            [ ''
              with lib;
              { options.foo = mkOption { }; }
              ''
            ]
        }
    }
