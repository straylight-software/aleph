let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "no-raw-writeshellapplication"
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
                                , regex = Some "^writeShellApplication$"
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
    , message = "ALEPH-W005: Use aleph.writeShellApplication"
    , note =
        ''
        ## What's wrong?
        Raw `writeShellApplication` was used instead of `aleph.writeShellApplication`.

        This is discouraged because it:
        - Doesn't integrate with aleph's conventions
        - May miss aleph-specific optimizations

        ## What can I do to fix this?
        Use `aleph.writeShellApplication` instead.
        ''
    , tests =
        { valid = [ "test" ]
        , invalid = [ "test" ]
        }
    }
