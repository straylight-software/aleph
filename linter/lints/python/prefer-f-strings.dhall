{- Requires f-strings instead of % formatting or .format().

f-strings are more readable and performant than % formatting
or .format() method calls.
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "python-prefer-f-strings"
    , language = "python"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "string"
      , regex = Some "^[%\"'].*%(s|d|f|\\.\\d+f|x|r|a)"
      }
    , message = "ALEPH-W040: Use f-strings instead of % formatting"
    , note =
      { description = "f-strings are more readable and performant than % formatting or .format()."
      , examples =
        [ "\"Hello, %s\" % name"
        , "\"Value: %d\" % value"
        , "\"Float: %.2f\" % pi"
        ]
      , suggested_fix =
          ''
          Use f-strings for readability and performance:

          ```python
          # BAD
          message = "Hello, %s" % name
          result = "Value: %d" % count
          formatted = "Pi: %.2f" % pi

          # GOOD
          message = f"Hello, {name}"
          result = f"Value: {count}"
          formatted = f"Pi: {pi:.2f}"
          ```
          ''
      }
    , tests =
      { valid =
        [ "f'Hello, {name}'"
        , "f'Value: {count}'"
        , "f'Pi: {pi:.2f}'"
        , "'literal string'"
        ]
      , extra_invalid = [] : List Text
      }
    }
