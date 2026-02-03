let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "rec-in-derivation"
    , language = "nix"
    , severity = Severity.Error
    , rule =
        { kind = "apply_expression"
        , regex = None Text
        , pattern = None Text
        , has = Some
            ( nodeMatcher
                { kind = Some "rec_attrset_expression"
                , field = Some "argument"
                , regex = None Text
                , has = None Schema.NodeMatcher
                , inside = None Schema.NodeMatcher
                }
            )
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-E002: `rec` used with mkDerivation"
    , note =
        ''
        ## What's wrong?
        A recursive attrset (`rec { ... }`) was passed to `mkDerivation`.

        This is forbidden because it:
        - Encourages self-referential derivation inputs
        - Obscures attribute evaluation order

        ## What can I do to fix this?
        Use the fixed-point form instead:

        ```nix
        stdenv.mkDerivation (finalAttrs: {
          # ...
        })
        ```
        ''
    , tests =
        { valid = 
            [ "stdenv.mkDerivation (finalAttrs: { name = \"foo\"; })"
            , "stdenv.mkDerivation { name = \"foo\"; }"
            ]
        , invalid = [ "stdenv.mkDerivation rec { name = \"foo\"; }" ]
        }
    }
