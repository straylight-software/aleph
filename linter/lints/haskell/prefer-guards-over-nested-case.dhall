{- Detects deeply nested case expressions that should use guards or where clauses.

From the Weyl Standard Haskell style guide:
> The Indentation Reality Check: In production code, deep nesting isn't just ugly—it's a maintenance liability.

The style guide recommends using guards with where clauses instead of deeply nested case expressions.

Examples:
  - BAD:  case x of A -> case y of B -> ... (deep nesting)
  - GOOD: process x | condition -> ... where condition = ... (flat guards)
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

let node = Schema.someNodeMatcher
let Match = Schema.NodeMatcherWithDefaults

in  { id = "haskell-prefer-guards-over-nested-case"
    , language = "haskell"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "case"
      , has =
          node
            Match::{
            , kind = Some "case"
            , stopBy = Some "end"
            }
      }
    , message = "ALEPH-H002: Deeply nested case expression - prefer guards"
    , note =
        ''
        ## What's wrong?
        Deeply nested case expressions are hard to read and maintain.

        From the Weyl Standard Haskell style guide:
        > The Indentation Reality Check: In production code, deep nesting isn't just ugly—it's a maintenance liability.

        Every level of indentation is a place where:
        - Merge conflicts multiply
        - Off-by-one space errors break compilation
        - Code reviews devolve into whitespace debates

        ## What can I do to fix this?
        Use guards with where clauses instead:

        ```haskell
        -- BAD: Philosophically pure but practically painful
        processRequest request =
          case validateRequest request of
            Nothing -> handleInvalid
            Just validReq ->
              case findRoute routes validReq of
                Nothing -> handleNoRoute
                Just route ->
                  case lookupHandler route of
                    Nothing -> handleMissingHandler
                    Just handler ->
                      executeHandler handler validReq

        -- GOOD: Flat is better than nested
        processRequest request = processValidated
          where
            processValidated
              | Nothing <- validateRequest request = handleInvalid
              | Just validReq <- validateRequest request = routeRequest validReq

            routeRequest validReq
              | Nothing <- findRoute routes validReq = handleNoRoute
              | Just route <- findRoute routes validReq = handleRoute route validReq

            handleRoute route validReq
              | Nothing <- lookupHandler route = handleMissingHandler
              | Just handler <- lookupHandler route = executeHandler handler validReq
        ```
        ''
    , tests =
        { valid = 
            [ "case x of Just y -> y"
            , "case x of A -> 1; B -> 2"
            , "process x | Just y <- x = y | otherwise = 0"
            , "foo x = case x of A -> 1; B -> 2"
            , "bar = case x of A -> 1"
            , "baz x = x + 1"
            ]
        , invalid = 
            [ "case x of Just y -> case y of Just z -> z"
            , "case x of A -> case y of B -> 1"
            , "case x of A -> case y of B -> case z of C -> 1"
            , "foo x = case x of A -> case y of B -> 1"
            , "bar x = case x of Just a -> case a of Just b -> b"
            , "baz x = case x of Left e -> case e of Error -> 1"
            ]
        }
    }
