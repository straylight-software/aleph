let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

let node = Schema.someNodeMatcher

let Match = Schema.NodeMatcherWithDefaults

in  { id = "rec-in-derivation"
    , language = "nix"
    , severity = Severity.Error
    , rule = Schema.Rule::{
      , kind = "apply_expression"
      , has =
          node
            Match::{
            , kind = Some "rec_attrset_expression"
            , field = Some "argument"
            }
      }
    , message = "ALEPH-E002: `rec` used with mkDerivation"
    , note =
        { description = "A recursive attrset (`rec { ... }`) was passed to `mkDerivation`."
        , examples =
          [ "stdenv.mkDerivation rec { name = \"foo\"; }"
          , "pkgs.stdenv.mkDerivation rec { pname = \"bar\"; version = \"1.0\"; }"
          , "mkDerivation rec { name = \"baz\"; }"
          ]
        , suggested_fix =
            ''
            Use the fixed-point form instead:

            ```nix
            stdenv.mkDerivation (finalAttrs: {
              # ...
            })
            ```
            ''
        }
    , tests =
      { valid =
        [ "stdenv.mkDerivation (finalAttrs: { name = \"foo\"; })"
        , "stdenv.mkDerivation { name = \"foo\"; }"
        , "pkgs.stdenv.mkDerivation { pname = \"bar\"; }"
        ]
      , extra_invalid = [] : List Text
      }
    }
