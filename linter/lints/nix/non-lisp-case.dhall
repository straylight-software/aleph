let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "non-lisp-case"
    , language = "nix"
    , severity = Severity.Warning
    , rule = Schema.Rule::{ kind = "identifier", regex = Some "[A-Z]" }
    , message = "ALEPH-W004: Use lisp-case for identifiers"
    , note =
        { description = "Identifier contains uppercase characters."
        , examples =
          [ "myFunction = 1"
          , "SomeValue = 1"
          , "IORef = 1"
          ]
        , suggested_fix =
            ''
            Use lisp-case (kebab-case) for all identifiers:

            ```nix
            my-function = ...;  # Good
            myFunction = ...;  # Bad
            ```
            ''
        }
    , tests =
      { valid = [ "my-function = 1", "my_function = 1", "lowercase = 1" ]
      , extra_invalid = [] : List Text
      }
    }
