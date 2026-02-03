{- Forbids using namespace directives.

The C++ style guide mandates fully qualified names for absolute clarity.
No `using namespace` declarations are allowed.

Examples:
  - BAD:  using namespace std;
  - BAD:  using namespace s4::core;
  - GOOD: std::vector, s4::core::result
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "cpp-no-using-namespace"
    , language = "cpp"
    , severity = Severity.Error
    , rule = Schema.Rule::{
      , kind = "using_declaration"
      , pattern = Some "using namespace $_"
      }
    , message = "ALEPH-E003: using namespace directive forbidden"
    , note =
        ''
        ## What's wrong?
        `using namespace` directives are forbidden. Use fully qualified names instead.

        ## Examples of violations:
        - `using namespace std;`
        - `using namespace s4::core;`
        - `using namespace boost;`

        ## What can I do to fix this?
        Use fully qualified names throughout:

        ```cpp
        // BAD
        using namespace std;
        vector<int> data;
        string name;

        // GOOD
        std::vector<int> data;
        std::string name;

        // BAD
        using namespace s4::core;
        result<int> compute();

        // GOOD
        s4::core::result<int> compute();
        ```

        ## Exceptions
        Type aliases for long names are encouraged:

        ```cpp
        // GOOD: Type alias for frequently-used type
        using inference_result = s4::inference::batch_result_t;
        ```
        ''
    , tests =
        { valid =
            [ "std::vector<int> data;"
            , "s4::core::result<int> compute();"
            , "using inference_result = s4::inference::batch_result_t;"
            , "namespace ns = s4::inference;"
            , "std::string name;"
            , "boost::optional<int> value;"
            ]
        , invalid =
            [ "using namespace std;"
            , "using namespace s4::core;"
            , "using namespace boost;"
            , "using namespace s4::inference;"
            , "using namespace std::chrono;"
            , "using namespace detail;"
            ]
        }
    }
