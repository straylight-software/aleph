{- Prefer newtype over type alias for domain boundaries and validation.

From the Weyl Standard Haskell style guide:
> Newtype Wrapping: Pragmatic Boundaries
> Always Wrap These:
> - Domain boundaries - prevents mixing up parameters
> - Units and semantics - when the type carries meaning
> - Validation boundaries - when construction can fail

Type aliases don't provide type safety, while newtypes do.
At -O2, GHC eliminates newtype overhead anyway.

Examples:
  - BAD:  type UserId = Int
  - GOOD: newtype UserId = UserId Int
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "haskell-prefer-newtype-over-type-alias"
    , language = "haskell"
    , severity = Severity.Info
    , rule = Schema.Rule::{
      , kind = "type_synomym"
      , regex = Some "(UserId|SessionId|RequestId|RouteId|Milliseconds|ByteCount|Percentage|Email|NodeId|TypeId|ScopeLevel)"
      }
    , message = "ALEPH-H005: Consider using newtype instead of type alias"
    , note =
        { description = "Type aliases don't provide type safety - they are just synonyms. This can lead to bugs where values of different semantic meanings are accidentally mixed up."
        , examples =
          [ "type UserId = Int"
          , "type SessionId = UUID"
          , "type RequestId = Int64"
          , "type Milliseconds = Int"
          , "type Email = Text"
          , "type RouteId = Text"
          ]
        , suggested_fix =
            ''
            From the Weyl Standard Haskell style guide:
            > Newtype Wrapping: Pragmatic Boundaries

            ## When to use newtype:
            - Domain boundaries (UserId, SessionId, RequestId, RouteId)
            - Units and semantics (Milliseconds, ByteCount, Percentage)
            - Validation boundaries (Email with smart constructor)
            - Compiler domain (NodeId, TypeId, ScopeLevel)

            Use newtype instead of type alias:

            ```haskell
            -- BAD: Type alias - no type safety
            type UserId = Int
            type SessionId = Int

            -- These compile but are semantically wrong:
            let userId = 42 :: UserId
            let sessionId = userId  -- Wrong! But compiles

            -- GOOD: Newtype - compiler enforces correctness
            newtype UserId = UserId { unUserId :: Int }
              deriving (Eq, Show)

            newtype SessionId = SessionId { unSessionId :: Int }
              deriving (Eq, Show)

            -- These are different types - compiler catches errors:
            let userId = UserId 42
            -- sessionId = userId  -- Compile error!
            ```

            With `-O2`, GHC eliminates newtype overhead completely.

            ## Rule: 
            Start with type aliases, upgrade to newtypes when you find bugs
            mixing things up.
            ''
        }
    , tests =
        { valid = 
            [ "newtype UserId = UserId Int"
            , "newtype SessionId = SessionId UUID"
            , "newtype Milliseconds = Milliseconds Int64"
            , "newtype Email = Email Text"
            , "type LoopCounter = Int"
            , "type CacheSize = Int"
            ]
        , extra_invalid = [] : List Text
        }
    }
