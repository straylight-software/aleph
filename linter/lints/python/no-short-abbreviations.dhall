{- Enforces the Three-Character Rule for identifiers.

If an identifier is 3 characters or less, it's probably wrong
for production code. Exceptions: idx/jdx, lhs/rhs, key/value, row/col
-}
let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "python-no-short-abbreviations"
    , language = "python"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "identifier"
      , regex = Some "^([a-zA-Z_][a-zA-Z0-9_]?[a-zA-Z0-9_]?)$"
      , not = Some
          { has = None Schema.NodeMatcher
          , inside = None Schema.NodeMatcher
          , regex = Some "^(idx|jdx|lhs|rhs|key|value|row|col)$"
          }
      }
    , message = "ALEPH-W044: Short identifier violates Three-Character Rule"
    , note =
      { description = "Identifiers with 3 or fewer characters create exponential confusion in agent-heavy codebases. Use full words instead."
      , examples =
        [ "cfg = load_config()"
        , "res = process(req)"
        , "ctx = get_context()"
        , "buf = allocate_buffer()"
        ]
      , suggested_fix =
          ''
          Use full words for clarity:

          ```python
          # BAD
          cfg = load_cfg()
          conn = mk_conn(cfg)
          res = proc(req)

          # GOOD
          configuration = load_model_configuration()
          connection = create_database_connection(configuration)
          result = process_inference_request(request)
          ```

          ## Standard exceptions (local scope only):
          - `idx`, `jdx` - indices in loops
          - `lhs`, `rhs` - left/right hand side
          - `key`, `value` - dictionary operations
          - `row`, `col` - matrix/grid operations
          ''
      }
    , tests =
      { valid =
        [ "configuration"
        , "connection"
        , "result"
        , "request"
        , "idx"
        , "lhs"
        , "key"
        , "row"
        ]
      , extra_invalid = [] : List Text
      }
    }
