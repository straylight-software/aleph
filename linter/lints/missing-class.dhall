let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "missing-class"
    , language = "nix"
    , severity = Severity.Info
    , rule =
        { kind = "apply_expression"
        , regex = None Text
        , pattern = None Text
        , has = Some
            ( nodeMatcher
                { kind = Some "identifier"
                , field = Some "function"
                , regex = Some "^mkDerivation$"
                , has = None Schema.NodeMatcher
                , inside = None Schema.NodeMatcher
                }
            )
        , not = Some
            { has = Some
                ( nodeMatcher
                    { kind = Some "binding"
                    , field = Some "binding"
                    , regex = Some "^pname$"
                    , has = None Schema.NodeMatcher
                    , inside = None Schema.NodeMatcher
                    }
                )
            , inside = None Schema.NodeMatcher
            }
        }
    , message = "ALEPH-I001: Missing pname (consider using mkDerivation with pname)"
    , note =
        ''
        ## What's wrong?
        `mkDerivation` is missing a `pname` attribute.

        This is informational because:
        - Using `pname` with `version` is the modern style
        - It allows automatic inference of the name attribute

        ## What can I do to fix this?
        Consider using `pname` and `version` instead of `name`:

        ```nix
        stdenv.mkDerivation (finalAttrs: {
          pname = "my-package";
          version = "1.0.0";
        })
        ```
        ''
    , tests =
        { valid = [ "mkDerivation { pname = \"foo\"; }" ]
        , invalid = [ "mkDerivation { name = \"foo\"; }" ]
        }
    }
