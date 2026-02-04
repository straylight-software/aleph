let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "missing-meta"
    , language = "nix"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "apply_expression"
      , regex = Some "mkDerivation"
      }
    , message = "ALEPH-W008: Missing meta attribute"
    , note =
        { description = "`mkDerivation` is missing a `meta` attribute."
        , examples =
          [ "mkDerivation { name = \"foo\"; }"
          , "mkDerivation { pname = \"bar\"; version = \"1.0\"; }"
          , "mkDerivation { src = ./.; }"
          ]
        , suggested_fix =
            ''
            Add a `meta` attribute with at least basic information.
            ''
        }
    , tests =
      { valid =
        [ "{ meta = {}; }"
        , "{ meta = { description = \"foo\"; }; }"
        , "{ meta = { license = lib.licenses.mit; }; }"
        ]
      , extra_invalid = [] : List Text
      }
    }
