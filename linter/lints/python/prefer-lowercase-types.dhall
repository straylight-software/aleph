{- Prefers lowercase type hints over capitalized ones.

Use `list[str]` instead of `List[str]`, `dict[str, int]` instead
of `Dict[str, int]`, etc. This is the modern Python 3.9+ way.
-}
let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "python-prefer-lowercase-types"
    , language = "python"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "identifier"
      , regex = Some "^(List|Dict|Set|Tuple)$"
      }
    , message = "ALEPH-W043: Use lowercase type hints (list, dict)"
    , note =
      { description = "Use lowercase type hints like `list[str]` instead of `List[str]` for modern Python 3.9+ compatibility."
      , examples =
        [ "List[str]"
        , "Dict[str, int]"
        , "Set[int]"
        , "Tuple[int, str]"
        ]
      , suggested_fix =
          ''
          Use lowercase type hints:

          ```python
          # BAD (old style)
          from typing import List, Dict, Set, Tuple

          def process(items: List[str]) -> Dict[str, int]:
              ...

          # GOOD (modern Python 3.9+)
          def process(items: list[str]) -> dict[str, int]:
              ...
          ```
          ''
      }
    , tests =
      { valid =
        [ "list[str]"
        , "dict[str, int]"
        , "set[int]"
        , "tuple[int, str]"
        ]
      , extra_invalid = [] : List Text
      }
    }
