let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "no-raw-mkderivation"
    , language = "nix"
    , severity = Severity.Warning
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
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-W007: Use aleph.mkDerivation"
    , note =
        ''
        ## What's wrong?
        Raw `mkDerivation` was used instead of `aleph.mkDerivation`.

        This is discouraged because it:
        - Doesn't integrate with aleph's conventions
        - May miss aleph-specific features

        ## What can I do to fix this?
        Use `aleph.mkDerivation` instead.
        ''
    , tests =
        { valid = [ "aleph.mkDerivation { name = \"foo\"; }" ]
        , invalid = [ "mkDerivation { name = \"foo\"; }" ]
        }
    }
