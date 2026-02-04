{- Requires type hints on all functions.

Every function should have explicit type annotations for
parameters and return types. Use `ty` in CI to enforce this.
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

let node = Schema.someNodeMatcher
let Match = Schema.NodeMatcherWithDefaults

in  { id = "python-missing-type-hints"
    , language = "python"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "function_definition"
      , not = Some
          { has = node Match::{ kind = Some "parameters", regex = Some ":" }
          , inside = None Schema.NodeMatcher
          , regex = None Text
          }
      }
    , message = "ALEPH-W046: Function missing type hints"
    , note =
      { description = "Every function must have explicit type annotations for parameters and return types. Type hints prevent runtime errors and serve as documentation."
      , examples =
        [ "def process(x):\n    return x * 2"
        , "def load_config(path):\n    with open(path) as f:\n        return json.load(f)"
        , "def calculate(a, b, c):\n    return a + b * c"
        ]
      , suggested_fix =
          ''
          Add explicit type annotations:

          ```python
          # BAD: Missing type hints
          def process(x):
              return x * 2

          def load_config(path):
              with open(path) as f:
                  return json.load(f)

          # GOOD: Explicit types
          def process(x: int) -> int:
              return x * 2

          def load_config(path: Path) -> dict[str, Any]:
              with open(path) as f:
                  return json.load(f)

          # GOOD: Complex types
          def process_batch(
              batches: list[Tensor],
              model: nn.Module,
              device: torch.device,
          ) -> list[Tensor]:
              ...
          ```
          ''
      }
    , tests =
      { valid =
        [ "def process(x: int) -> int:\n    return x * 2"
        , "def load_config(path: Path) -> dict[str, Any]:\n    ..."
        , "def calculate(a: float, b: float, c: float) -> float:\n    return a + b * c"
        ]
      , extra_invalid = [] : List Text
      }
    }
