{- Requires enum class (scoped enums) instead of plain enum.

Use scoped enums (enum class) instead of unscoped enums to prevent
name pollution and implicit conversions.

Examples:
  - BAD:  enum color { red, green, blue };
  - GOOD: enum class color { red, green, blue };
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "cpp-enum-class"
    , language = "cpp"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "enum_specifier"
      , pattern = Some "enum $_"
      , not = Some
          { has = None Schema.NodeMatcher
          , inside = None Schema.NodeMatcher
          , regex = Some "enum\\s+class"
          }
      }
    , message = "ALEPH-W011: Use enum class instead of plain enum"
    , note =
        { description = "Use scoped enums (enum class) instead of unscoped enums."
        , examples =
          [ "enum color { red, green, blue };"
          , "enum status { ok, error };"
          , "enum type { int_, float_ };"
          , "enum direction { north, south };"
          , "enum state { active, inactive };"
          ]
        , suggested_fix =
            ''
            Use enum class:

            ```cpp
            // BAD
            enum color { red, green, blue };
            enum status { ok, error, warning };

            // GOOD
            enum class color { red, green, blue };
            enum class status { ok, error, warning };

            // Usage
            auto c = color::red;  // Scoped access
            if (s == status::ok) { ... }
            ```

            ## Benefits
            - No name pollution in surrounding scope
            - No implicit conversion to int
            - Type safety: must use scope operator (::)
            ''
        }
    , tests =
        { valid = 
            [ "enum class color { red, green, blue };"
            , "enum class status { ok, error };"
            , "enum class type { int_, float_ };"
            , "enum class direction { north, south };"
            , "enum class state { active, inactive };"
            ]
        , extra_invalid = [] : List Text
        }
    }
