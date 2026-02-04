{- Discourages deeply nested case expressions (3+ levels).

Deep nesting creates maintenance liabilities:
- Merge conflicts multiply
- Off-by-one space errors break compilation
- Code reviews devolve into whitespace debates
- Even good IDEs struggle with layout rules

Prefer guards or extracted functions for complex control flow.
-}
let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "haskell-no-deep-nesting"
    , language = "haskell"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "case"
      , regex = Some "(?s)case.*of.*case.*of.*case.*of"
      }
    , message = "ALEPH-W025: Deeply nested case expressions (3+ levels)"
    , note =
      { description =
          "Case expressions nested 3+ levels deep create maintenance liabilities and compilation issues."
      , examples =
        [ ''
          case x of
            Just y -> case y of
              Just z -> case z of
                Just w -> w
                Nothing -> 0
              Nothing -> 0
            Nothing -> 0''
        , ''
          processRequest request = case validateRequest request of
            Just validReq -> case findRoute routes validReq of
              Just route -> case lookupHandler route of
                Just handler -> executeHandler handler validReq
                Nothing -> handleMissingHandler''
        , ''
          parseResult input = case input of
            Left err -> case err of
              ParseError msg -> case msg of
                "syntax" -> SyntaxError
                _ -> UnknownError
            Right val -> Success val''
        ]
      , suggested_fix =
          ''
          Flatten with guards or extract to separate functions:

          ```haskell
          -- BAD: Deep nesting
          processRequest request =
            case validateRequest request of
              Just validReq ->
                case findRoute routes validReq of
                  Just route ->
                    case lookupHandler route of
                      Just handler -> executeHandler handler validReq
                      Nothing -> handleMissingHandler

          -- GOOD: Use guards with where clause
          processRequest request = processValidated
            where
              processValidated
                | Just validReq <- validateRequest request = routeRequest validReq
                | otherwise = handleInvalid

              routeRequest validReq
                | Just route <- findRoute routes validReq = handleRoute route validReq
                | otherwise = handleNoRoute
          ```
          ''
      }
    , tests =
      { valid =
        [ "case x of Just y -> y; Nothing -> 0"
        , ''
          case x of
            Just y -> case y of
              Just z -> z
              Nothing -> 0
            Nothing -> 0''
        , ''
          processRequest request
            | Just validReq <- validateRequest request = handleValid validReq
            | otherwise = handleInvalid''
        ]
      , extra_invalid = [] : List Text
      }
    }
