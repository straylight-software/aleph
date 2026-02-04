let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "missing-class"
    , language = "nix"
    , severity = Severity.Error
    , rule = Schema.Rule::{ kind = "attrset_expression", regex = Some "^\\{" }
    , message = "ALEPH-E004: missing `_class` attribute"
    , note =
        { description = "Module files must define `_class`."
        , examples =
          [ "{ config = {}; }"
          , "{ options = {}; config = {}; }"
          , "{ imports = []; }"
          ]
        , suggested_fix =
            ''
            Add `_class` to the module attrset.
            ''
        }
    , tests =
      { valid = [ "[]", "\"string\"", "123" ]
      , extra_invalid = [] : List Text
      }
    }
