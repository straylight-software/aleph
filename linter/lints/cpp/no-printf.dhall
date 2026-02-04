{- Forbids printf-style functions in favor of std::format.

Use std::format (C++20) or iostream instead of printf/fprintf/sprintf
for type-safe formatting.

Examples:
  - BAD:  printf("Hello %s\n", name);
  - GOOD: std::cout << "Hello " << name << std::endl;
  - GOOD: std::format("Hello {}\n", name);
-}
let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "cpp-no-printf"
    , language = "cpp"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "call_expression"
      , pattern = Some "printf(\$\$\$ARGS)"
      }
    , message = "ALEPH-W020: Use std::format or iostreams instead of printf"
    , note =
        ''
        ## What's wrong?
        printf is not type-safe and can cause runtime errors. Use modern C++ alternatives.

        ## Examples of violations:
        - `printf("Hello %s\n", name);`
        - `fprintf(stderr, "Error: %d\n", code);`
        - `sprintf(buffer, "%s", str);`

        ## What can I do to fix this?
        Use type-safe alternatives:

        ```cpp
        // BAD: printf
        printf("Hello %s, you have %d messages\n", name, count);

        // GOOD: iostreams
        std::cout << "Hello " << name << ", you have " << count << " messages\n";

        // GOOD: std::format (C++20)
        auto msg = std::format("Hello {}, you have {} messages\n", name, count);
        std::cout << msg;
        ```

        ## Benefits
        - Type safety: Compile-time type checking
        - No format string vulnerabilities
        - Modern C++ style
        - Better performance with std::format
        ''
    , tests =
      { valid =
        [ "std::cout << name << std::endl;"
        , "std::format(\"Hello {}\\n\", name);"
        , "std::println(\"Hello {}\", name);"
        , "fmt::format(\"Hello {}\\n\", name);"
        , "std::cerr << error << std::endl;"
        ]
      , invalid =
        [ "printf(\"Hello %s\\n\", name);"
        , "printf(\"Count: %d\\n\", count);"
        , "printf(\"%.2f\\n\", value);"
        ]
      }
    }
