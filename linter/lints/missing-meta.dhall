let Schema = ../schemas/Lint.dhall
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
        ''
        ## What's wrong?
        `mkDerivation` is missing a `meta` attribute.

        ## What can I do to fix this?
        Add a `meta` attribute with at least basic information.
        ''
    , tests =
        { valid = [ "{}" ]
        , invalid = [ "mkDerivation { name = \"foo\"; }" ]
        }
    }
