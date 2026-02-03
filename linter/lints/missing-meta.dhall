let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "missing-meta"
    , language = "nix"
    , severity = Severity.Warning
    , rule =
        { kind = "apply_expression"
        , regex = None Text
        , pattern = None Text
        , has = Some
            ( nodeMatcher
                { kind = Some "variable_expression"
                , field = Some "function"
                , regex = None Text
                , has = Some
                    ( nodeMatcher
                        { kind = Some "identifier"
                        , field = None Text
                        , regex = Some "^mkDerivation$"
                        , has = None Schema.NodeMatcher
                        , inside = None Schema.NodeMatcher
                        }
                    )
                , inside = None Schema.NodeMatcher
                }
            )
        , not = Some
            { has = Some
                ( nodeMatcher
                    { kind = Some "attrset_expression"
                    , field = Some "argument"
                    , regex = None Text
                    , has = Some
                        ( nodeMatcher
                            { kind = Some "attrpath"
                            , field = None Text
                            , regex = Some "^meta$"
                            , has = None Schema.NodeMatcher
                            , inside = None Schema.NodeMatcher
                            }
                        )
                    , inside = None Schema.NodeMatcher
                    }
                )
            , inside = None Schema.NodeMatcher
            }
        }
    , message = "ALEPH-W008: Missing meta attribute"
    , note =
        ''
        ## What's wrong?
        `mkDerivation` is missing a `meta` attribute.

        This is discouraged because it:
        - Omits important package metadata
        - Makes it harder to find information about the package

        ## What can I do to fix this?
        Add a `meta` attribute with at least basic information:

        ```nix
        meta = {
          description = "Brief description";
          license = lib.licenses.mit;
        };
        ```
        ''
    , tests =
        { valid =
            [ ''
              stdenv.mkDerivation { meta = { description = "foo"; }; }
              ''
            ]
        , invalid =
            [ ''
              stdenv.mkDerivation { name = "foo"; }
              ''
            ]
        }
    }
