let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "missing-description"
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
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-W009: Missing meta.description"
    , note =
        ''
        ## What's wrong?
        `mkDerivation` is missing a `meta.description` attribute.

        This is discouraged because it:
        - Makes it harder to understand what the package does
        - Is required for many nixpkgs contributions

        ## What can I do to fix this?
        Add a `meta.description` attribute:

        ```nix
        meta = {
          description = "Brief description of what this package does";
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
              stdenv.mkDerivation { meta = { license = lib.licenses.mit; }; }
              ''
            ]
        }
    }
