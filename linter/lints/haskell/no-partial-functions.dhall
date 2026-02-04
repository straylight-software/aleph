{- Discourages use of partial functions.

Partial functions like head, tail, fromJust can fail at runtime
and should be avoided in production code. Use total alternatives
like pattern matching, uncons, or proper error handling.
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "haskell-no-partial-functions"
    , language = "haskell"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "variable"
      , regex = Some "^(head|tail|init|last|fromJust|fromLeft|fromRight|read)$"
      }
    , message = "ALEPH-W027: Avoid partial functions"
    , note =
      { description = "Partial functions can fail at runtime. Use pattern matching or total alternatives instead."
      , examples =
        [ "head xs"
        , "tail xs"
        , "fromJust maybeValue"
        , "read inputString"
        ]
      , suggested_fix =
          ''
          Use total alternatives:

          ```haskell
          -- BAD: Partial functions
          firstElement = head xs
          rest = tail xs
          value = fromJust maybeValue
          parsed = read input

          -- GOOD: Pattern matching
          firstElement = case xs of
            (x:_) -> Just x
            []    -> Nothing

          -- GOOD: Use uncons
          firstElement = fst <$> uncons xs

          -- GOOD: Explicit error handling
          value = case maybeValue of
            Just v  -> v
            Nothing -> defaultValue

          -- GOOD: Safe parsing with reads
          case reads input of
            [(parsed, "")] -> Right parsed
            _              -> Left "Parse error"
          ```
          ''
      }
    , tests =
      { valid =
        [ "case xs of (x:_) -> Just x; [] -> Nothing"
        , "uncons xs"
        , "fromMaybe defaultValue maybeValue"
        , "reads input"
        ]
      , extra_invalid = [] : List Text
      }
    }
