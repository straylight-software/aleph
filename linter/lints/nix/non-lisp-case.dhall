let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "non-lisp-case"
    , language = "nix"
    , severity = Severity.Warning
    , rule = Schema.Rule::{ kind = "identifier", regex = Some "[A-Z]" }
    , message = "ALEPH-W004: Use lisp-case for identifiers"
    , note =
        ''
        ## What's wrong?
        Identifier contains uppercase characters.

        ## What can I do to fix this?
        Use lisp-case (kebab-case) for all identifiers:

        ```nix
        my-function = ...;  # Good
        myFunction = ...;  # Bad
        ```
        ''
    , tests =
      { valid = [ "my-function = 1", "my_function = 1", "lowercase = 1" ]
      , invalid = [ "myFunction = 1", "SomeValue = 1", "IORef = 1" ]
      }
    }
