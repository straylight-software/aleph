let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "non-lisp-case"
    , language = "nix"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "identifier"
        , regex = Some "[A-Z]"
        , pattern = None Text
        , has = None Schema.NodeMatcher
        , not = None Schema.RuleNot
        }
    , message = "ALEPH-W004: Use lisp-case for identifiers"
    , note =
        ''
        ## What's wrong?
        Identifier contains uppercase characters.

        This is discouraged because it:
        - Breaks consistency with nixpkgs conventions
        - Makes it harder to remember the exact name

        ## What can I do to fix this?
        Use lisp-case (kebab-case) for all identifiers:

        ```nix
        my-function = ...;  # Good
        myFunction = ...;  # Bad
        ```
        ''
    , tests =
        { valid =
            [ "my-function = 1"
            , "my_function = 1"
            ]
        , invalid =
            [ "myFunction = 1"
            ]
        }
    }
