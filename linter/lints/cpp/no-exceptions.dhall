{- Forbids throw statements and try-catch blocks.

The C++ style guide mandates using s4::core::result<T> for error handling
instead of exceptions. This makes error handling explicit and visible.

Examples:
  - BAD:  throw std::runtime_error("...");
  - BAD:  try { ... } catch (...) { ... }
  - GOOD: return s4::fail<T>("...");
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "cpp-no-exceptions"
    , language = "cpp"
    , severity = Severity.Error
    , rule = Schema.Rule::{
      , kind = "throw_statement"
      , pattern = Some "throw $_"
      }
    , message = "ALEPH-E004: throw statements forbidden, use s4::core::result<T>"
    , note =
        ''
        ## What's wrong?
        Exception handling with `throw` is forbidden. Use `s4::core::result<T>` instead.

        ## Examples of violations:
        - `throw std::runtime_error("...");`
        - `throw std::invalid_argument("...");`
        - `throw my_custom_exception();`

        ## What can I do to fix this?
        Use result types for error handling:

        ```cpp
        // BAD
        auto parse_config(const std::string& json) -> configuration {
          if (json.empty()) {
            throw std::invalid_argument("empty json");
          }
          return configuration{...};
        }

        // GOOD
        auto parse_config(std::string_view json)
          -> s4::core::result<configuration> {
          if (json.empty()) {
            return s4::fail<configuration>("empty json");
          }
          return s4::ok(configuration{...});
        }
        ```

        ## Handling errors
        Check results explicitly:

        ```cpp
        auto result = parse_config(json);
        if (!result) {
          s4::error("failed to parse: {}", result.error().what());
          return;
        }
        auto config = *result;
        ```

        ## Fatal errors
        For truly unrecoverable errors:

        ```cpp
        if (!critical_resource) {
          s4::fatal("critical resource unavailable: {}", resource_name);
        }
        ```
        ''
    , tests =
        { valid =
            [ "return s4::ok(value);"
            , "return s4::fail<config>(\"error\");"
            , "auto result = parse_config();"
            , "if (!result) { return; }"
            , "s4::fatal(\"critical error\");"
            , "return s4::core::ok();"
            ]
        , invalid =
            [ "throw std::runtime_error(\"...\");"
            , "throw std::invalid_argument(\"...\");"
            , "throw my_exception();"
            , "throw 42;"
            , "throw \"error\";"
            , "throw std::exception();"
            ]
        }
    }
