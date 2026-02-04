{- Requires double quotes for strings.

Per Weyl Standard Python: use double quotes consistently.
-}
let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "python-no-single-quotes"
    , language = "python"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "string"
      , regex = Some "^'[^']*'$"
      }
    , message = "ALEPH-W042: Use double quotes for strings"
    , note =
      { description = "Per Weyl Standard Python, use double quotes for all strings consistently."
      , examples =
        [ "'hello world'"
        , "'single quoted'"
        , "'another string'"
        ]
      , suggested_fix =
          ''
          Use double quotes consistently:

          ```python
          # BAD
          name = 'John'
          message = 'Hello, world!'

          # GOOD
          name = "John"
          message = "Hello, world!"
          ```
          ''
      }
    , tests =
      { valid =
        [ "hello world"
        , "double quoted"
        , "another string"
        , "yet another"
        ]
      , extra_invalid = [] : List Text
      }
    }
