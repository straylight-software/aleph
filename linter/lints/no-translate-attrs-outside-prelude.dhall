let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "no-translate-attrs-outside-prelude"
    , language = "nix"
    , severity = Severity.Error
    , rule =
        { kind = "select_expression"
        , regex = None Text
        , pattern = None Text
        , has = Some
            ( nodeMatcher
                { kind = Some "attrpath"
                , field = Some "attrpath"
                , regex = None Text
                , has = Some
                    ( nodeMatcher
                        { kind = Some "identifier"
                        , field = None Text
                        , regex = Some "^translateAttrs$"
                        , has = None Schema.NodeMatcher
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
    , message = "ALEPH-E003: translateAttrs should only be used in prelude"
    , note =
        ''
        ## What's wrong?
        `translateAttrs` was used outside of the prelude.

        This is forbidden because it:
        - Is an internal prelude function
        - Should not be used in user code

        ## What can I do to fix this?
        Use standard attribute manipulation functions instead.
        ''
    , tests =
        { valid = [ "test" ]
        , invalid = [ "test" ]
        }
    }
