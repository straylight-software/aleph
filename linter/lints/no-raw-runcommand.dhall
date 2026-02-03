let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "no-raw-runcommand"
    , language = "nix"
    , severity = Severity.Warning
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
                                , regex = Some "^runCommand$"
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
        , not = Some
            { has = None Schema.NodeMatcher
            , inside = Some
                ( nodeMatcher
                    { kind = Some "select_expression"
                    , field = None Text
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
                                    , regex = Some "^aleph$"
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
            }
        }
    , message = "ALEPH-W006: Use aleph.runCommand"
    , note =
        ''
        ## What's wrong?
        Raw `runCommand` was used instead of `aleph.runCommand`.

        This is discouraged because it:
        - Doesn't integrate with aleph's conventions
        - May miss aleph-specific features

        ## What can I do to fix this?
        Use `aleph.runCommand` instead.
        ''
    , tests =
        { valid = [ "test" ]
        , invalid = [ "test" ]
        }
    }
