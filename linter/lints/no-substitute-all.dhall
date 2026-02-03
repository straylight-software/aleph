let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "no-substitute-all"
    , language = "nix"
    , severity = Severity.Error
    , rule =
        { kind = "apply_expression"
        , regex = None Text
        , pattern = None Text
        , has = Some
            ( nodeMatcher
                { kind = Some "select_expression"
                , field = Some "function"
                , regex = None Text
                , has = Some
                    ( nodeMatcher
                        { kind = Some "attrpath"
                        , field = Some "attrpath"
                        , regex = None Text
                        , has = Some
                            ( nodeMatcher
                                { kind = Some "identifier"
                                , field = None Text
                                , regex = Some "^substituteAll$"
                                , has = None Schema.NodeMatcher
                                , inside = None Schema.NodeMatcher
                                }
                            )
                        , inside = None Schema.NodeMatcher
                        }
                    )
                , inside = None Schema.NodeMatcher
                }
            )
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-E004: Avoid using substituteAll"
    , note =
        ''
        ## What's wrong?
        `substituteAll` was used.

        This is forbidden because it:
        - Is fragile and hard to debug
        - Makes code harder to understand

        ## What can I do to fix this?
        Use explicit string interpolation or structured arguments instead.
        ''
    , tests =
        { valid =
            [ "substitute { src = ./file; }"
            ]
        , invalid =
            [ "substituteAll { src = ./file; }"
            ]
        }
    }
