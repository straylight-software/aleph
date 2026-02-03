{- Forbids C-style casts in favor of C++ casts.

Use C++ style casts (static_cast, dynamic_cast, const_cast, reinterpret_cast)
instead of C-style casts. C++ casts are more explicit and searchable.

Examples:
  - BAD:  (int)x
  - GOOD: static_cast<int>(x)
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "cpp-no-c-style-cast"
    , language = "cpp"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "cast_expression"
      , regex = Some "^\\("
      }
    , message = "ALEPH-W012: Use C++ style cast instead of C-style cast"
    , note =
        ''
        ## What's wrong?
        C-style casts are forbidden. Use C++ style casts for clarity and safety.

        ## Examples of violations:
        - `(int)x`
        - `(float)y`
        - `(void*)ptr`
        - `(const char*)data`

        ## What can I do to fix this?
        Use appropriate C++ cast:

        ```cpp
        // BAD: C-style casts
        auto i = (int)f;
        auto p = (void*)data;
        auto base = (base_class*)derived;

        // GOOD: C++ style casts
        auto i = static_cast<int>(f);
        auto p = reinterpret_cast<void*>(data);
        auto base = dynamic_cast<base_class*>(derived);
        auto ptr = const_cast<char*>(data);
        ```

        ## Cast selection guide
        - `static_cast`: Safe conversions (int to float, derived to base)
        - `dynamic_cast`: Polymorphic downcasts (with runtime check)
        - `const_cast`: Remove const/volatile (avoid if possible)
        - `reinterpret_cast`: Low-level reinterpreting (dangerous, avoid)
        ''
    , tests = 
        { valid = 
            [ "static_cast<int>(f)"
            , "dynamic_cast<base*>(derived)"
            , "reinterpret_cast<void*>(ptr)"
            , "const_cast<char*>(data)"
            , "static_cast<float>(i)"
            , "static_cast<size_t>(count)"
            ]
        , invalid = 
            [ "(int)f"
            , "(float)x"
            , "(void*)ptr"
            , "(char*)data"
            , "(size_t)count"
            , "(double)value"
            ]
        }
    }
