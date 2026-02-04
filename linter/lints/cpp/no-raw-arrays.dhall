{- Discourages C-style raw arrays in favor of std::array/std::vector.

Use std::array for fixed-size arrays or std::vector for dynamic arrays
instead of C-style arrays. They provide bounds checking, automatic
memory management, and standard container interface.

Examples:
  - BAD:  int arr[10];
  - GOOD: std::array<int, 10> arr;
  - GOOD: std::vector<int> arr(10);
-}
let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "cpp-no-raw-arrays"
    , language = "cpp"
    , severity = Severity.Warning
    , rule = Schema.Rule::{ kind = "array_declarator" }
    , message =
        "ALEPH-W022: Use std::array or std::vector instead of C-style array"
    , note =
        { description = "C-style arrays decay to pointers, have no bounds checking, and don't support modern container operations."
        , examples =
          [ "int arr[10];"
          , "char buffer[256];"
          , "double matrix[10][10];"
          , "int values[] = {1, 2, 3};"
          , "int matrix2[5][5][5];"
          ]
        , suggested_fix =
            ''
            Use standard containers:

            ```cpp
            // BAD: C-style arrays
            int arr[10];
            char buffer[256];
            int matrix[10][10];

            // Accessing is unsafe
            arr[15] = 5;  // Undefined behavior, no bounds check

            // GOOD: std::array for fixed size
            std::array<int, 10> arr;
            std::array<char, 256> buffer;
            std::array<std::array<int, 10>, 10> matrix;

            // Bounds-checked access
            arr.at(5) = 10;  // Throws if out of bounds
            if (arr.size() > 5) arr[5] = 10;  // Safe

            // GOOD: std::vector for dynamic size
            std::vector<int> vec(10);
            vec.push_back(42);
            ```

            ## Benefits
            - Bounds checking with .at()
            - No array-to-pointer decay
            - Standard container interface (begin, end, size)
            - Works with STL algorithms
            - Type-safe
            ''
        }
    , tests =
      { valid =
        [ "std::array<int, 10> arr;"
        , "std::vector<int> vec(10);"
        , "std::array<char, 256> buffer;"
        , "std::vector<double> matrix(100);"
        , "int* ptr = new int[10];"
        , "const char* str = \"hello\";"
        ]
      , extra_invalid = [] : List Text
      }
    }
