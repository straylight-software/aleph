{- Discourages use of "red light" language extensions.

These extensions are flagged as requiring justification in the Weyl Haskell style guide:
- DataKinds - Type-level programming rarely pays off in apps
- TypeOperators - Compile times and error messages suffer
- UndecidableInstances - Usually means you're solving the wrong problem
- ImplicitParams - Debugging nightmare
- OverlappingInstances - Semantic timebomb

Prefer the "green light" extensions instead.
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "haskell-no-red-light-extensions"
    , language = "haskell"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "pragma"
      , regex = Some "(DataKinds|TypeOperators|UndecidableInstances|ImplicitParams|OverlappingInstances)"
      }
    , message = "ALEPH-W024: Red light extension requires justification"
    , note =
        { description = "This language extension is in the 'red light' category and requires strong justification per Weyl Haskell style guide."
        , examples =
          [ "{-# LANGUAGE DataKinds #-}"
          , "{-# LANGUAGE TypeOperators #-}"
          , "{-# LANGUAGE UndecidableInstances #-}"
          ]
        , suggested_fix =
            ''
            Consider using 'green light' extensions instead:

            ```haskell
            -- Green light - use freely
            {-# LANGUAGE BangPatterns #-}
            {-# LANGUAGE OverloadedStrings #-}
            {-# LANGUAGE RecordWildCards #-}
            {-# LANGUAGE DeriveGeneric #-}
            {-# LANGUAGE StrictData #-}
            ```

            If you must use a red light extension, document why:
            ```haskell
            -- human: Needed for type-safe routing in servant
            -- human: Compile-time impact acceptable for this use case
            {-# LANGUAGE DataKinds #-}
            ```
            ''
        }
    , tests =
      { valid =
        [ "{-# LANGUAGE BangPatterns #-}"
        , "{-# LANGUAGE OverloadedStrings #-}"
        , "{-# LANGUAGE RecordWildCards #-}"
        , "{-# LANGUAGE StrictData #-}"
        , "{-# LANGUAGE DeriveGeneric #-}"
        ]
      , extra_invalid =
        [ "{-# LANGUAGE ImplicitParams #-}"
        , "{-# LANGUAGE OverlappingInstances #-}"
        ]
      }
    }
