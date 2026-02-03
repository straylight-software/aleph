let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "missing-description"
    , language = "nix"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "apply_expression"
      , regex = Some "mkOption"
      }
    , message = "ALEPH-W005: missing description attribute"
    , note =
        ''
        ## What's wrong?
        `mkOption` was called without a `description` attribute.

        ## What can I do to fix this?
        Add a human-readable `description` for the option.
        ''
    , tests =
        { valid = [ "{}" ]
        , invalid = [ "mkOption { type = types.str; }" ]
        }
    }
