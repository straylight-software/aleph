{- Discourages use of auto in favor of explicit types.

The C++ style guide prefers explicit types over auto for clarity,
especially in agent-heavy development where type inference can hide
important information.

Examples:
  - DISCOURAGED: auto config = load_config();
  - PREFERRED: configuration config = load_configuration();

Note: This is a warning, not an error, as auto is sometimes acceptable.
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "cpp-explicit-types"
    , language = "cpp"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "declaration"
      , regex = Some "^\\s*auto\\s+"
      }
    , message = "ALEPH-W008: Prefer explicit types over auto"
    , note =
        { description = "`auto` can hide type information. Prefer explicit types for clarity."
        , examples =
          [ "auto config = load();"
          , "auto count = 42;"
          , "auto name = \"foo\";"
          , "auto result = compute();"
          , "auto conn = get_connection();"
          ]
        , suggested_fix =
            ''
            Use explicit types when the type conveys semantic meaning:

            ```cpp
            // DISCOURAGED
            auto config = load_configuration();
            auto result = process_data(input);
            auto conn = get_connection();

            // PREFERRED
            configuration config = load_configuration();
            process_result result = process_data(input);
            database_connection conn = get_connection();
            ```

            ## When auto is acceptable
            - Complex iterator types: `auto it = container.begin();`
            - Lambda types: `auto callback = [&] { ... };`
            - Type is obvious from context: `auto x = static_cast<int>(y);`

            ## Guideline
            Use explicit types when the type conveys semantic meaning.
            Use auto when the type is verbose and obvious from context.
            ''
        }
    , tests =
        { valid = 
            [ "configuration config = load();"
            , "int count = 42;"
            , "std::string name = \"foo\";"
            , "result_t result = compute();"
            , "connection conn = get_connection();"
            ]
        , extra_invalid = [] : List Text
        }
    }
