{- Enforces snake_case naming convention for all identifiers.

The C++ style guide mandates snake_case for everything - variables, functions,
classes, constants, etc. This eliminates ambiguity about naming conventions
and makes code globally searchable.

Examples:
  - BAD:  myFunction, MyClass, myVariable, HTTPRequest, parseJSON
  - GOOD: my_function, my_class, my_variable, http_request, parse_json
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "cpp-snake-case"
    , language = "cpp"
    , severity = Schema.Severity.Error
    , rule = Schema.Rule::{
      , kind = "identifier"
      , regex = Some "[A-Z]"
      }
    , message = "ALEPH-E002: Identifier must use snake_case (contains uppercase)"
    , note =
        { description = "C++ identifiers must use snake_case (lowercase with underscores)."
        , examples =
          [ "myFunction"
          , "MyClass"
          , "MyVariable"
          , "processData"
          , "getValue"
          , "HTTPRequest"
          , "parseJSON"
          , "tensorBatch"
          , "DeviceMemory"
          , "Configuration"
          , "getID"
          ]
        , suggested_fix =
            ''
            Convert camelCase and PascalCase to snake_case:

            ```cpp
            // BAD
            class MyClass {
              void processData();
              int getValue();
            };
            auto myVariable = 42;

            // GOOD
            class my_class {
              void process_data();
              int get_value();
            };
            auto my_variable = 42;
            ```

            ## Note on acronyms
            Acronyms should be lowercase: `http_request` not `HTTPRequest`,
            `parse_json` not `parseJSON`, `gpu_memory` not `GPUMemory`.
            ''
        }
    , tests =
        { valid =
            [ "my_function"
            , "my_class"
            , "my_variable"
            , "process_data"
            , "get_value"
            , "http_request"
            , "parse_json"
            , "tensor_batch"
            , "device_memory"
            , "configuration"
            , "connection"
            ]
        , extra_invalid = [] : List Text
        }
    }
