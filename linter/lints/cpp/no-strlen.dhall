{- Forbids strlen in favor of std::string::size().

Use std::string and its member functions instead of C-style
string functions like strlen, strcpy, strcat.

Examples:
  - BAD:  strlen(name)
  - GOOD: name.size()
-}
let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "cpp-no-strlen"
    , language = "cpp"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "call_expression"
      , pattern = Some "strlen(\$ARG)"
      }
    , message = "ALEPH-W021: Use std::string::size() instead of strlen"
    , note =
        ''
        ## What's wrong?
        strlen and other C string functions are unsafe. Use std::string for automatic
        memory management and bounds checking.

        ## Examples of violations:
        - `strlen(name)`
        - `strlen(buffer)`
        - `if (strlen(str) > 0)`

        ## What can I do to fix this?
        Use std::string member functions:

        ```cpp
        // BAD: C-style strings
        char buffer[100];
        strcpy(buffer, input);
        size_t len = strlen(buffer);
        if (len > 0) {
            strcat(buffer, "suffix");
        }

        // GOOD: std::string
        std::string buffer = input;
        if (!buffer.empty()) {
            buffer += "suffix";
        }
        size_t len = buffer.size();
        ```

        ## Benefits
        - Automatic memory management
        - No buffer overflows
        - Modern C++ style
        - Self-documenting code
        ''
    , tests =
      { valid =
        [ "name.size();"
        , "str.length();"
        , "if (!buffer.empty())"
        , "std::string s = input;"
        , "s += \"suffix\";"
        ]
      , invalid =
        [ "strlen(name);"
        , "strlen(buffer.c_str());"
        , "if (strlen(str) > 0)"
        , "size_t len = strlen(input);"
        , "while (strlen(p) < 10)"
        ]
      }
    }
