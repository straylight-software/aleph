let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "missing-class"
    , language = "nix"
    , severity = Severity.Error
    , rule = Schema.Rule::{ kind = "attrset_expression", regex = Some "^\\{" }
    , message = "ALEPH-E004: missing `_class` attribute"
    , note =
        ''
        ## What's wrong?
        Module files must define `_class`.

        ## What can I do to fix this?
        Add `_class` to the module attrset.
        ''
    , tests =
      { valid = [ "[]", "\"string\"", "123" ]
      , invalid =
        [ "{ config = {}; }"
        , "{ options = {}; config = {}; }"
        , "{ imports = []; }"
        ]
      }
    }
