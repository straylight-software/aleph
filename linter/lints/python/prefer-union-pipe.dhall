{- Prefers union pipe syntax over Optional and Union.

Use `str | None` instead of `Optional[str]` or `Union[str, None]`.
This is more readable and is the modern Python 3.10+ way.
-}
let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "python-prefer-union-pipe"
    , language = "python"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "identifier"
      , regex = Some "^(Optional|Union)$"
      }
    , message = "ALEPH-W041: Use union pipe syntax (|) instead of Optional/Union"
    , note =
      { description = "Use union pipe syntax `str | None` instead of `Optional[str]` for better readability."
      , examples =
        [ "Optional[str]"
        , "Optional[int]"
        , "Union[str, int]"
        , "Union[str, int, None]"
        ]
      , suggested_fix =
          ''
          Use union pipe syntax for cleaner type annotations:

          ```python
          # BAD
          def find_user(user_id: int) -> Optional[User]:
              ...

          def process(value: Union[str, int]) -> Result:
              ...

          # GOOD
          def find_user(user_id: int) -> User | None:
              ...

          def process(value: str | int) -> Result:
              ...
          ```
          ''
      }
    , tests =
      { valid =
        [ "str | None"
        , "int | str"
        , "User | None"
        , "list[str | int]"
        ]
      , extra_invalid = [] : List Text
      }
    }
