{- Discourages use of error and undefined.

Runtime exceptions via error and undefined subvert the type system
and make code harder to reason about. Use proper error handling
like Either, Maybe, or custom exception types instead.
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "haskell-no-error-undefined"
    , language = "haskell"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "variable"
      , regex = Some "^(error|undefined)$"
      }
    , message = "ALEPH-W026: Use proper error handling instead of error/undefined"
    , note =
      { description = "Runtime exceptions via error and undefined subvert the type system. Use Either, Maybe, or custom exceptions."
      , examples =
        [ "error \"Not implemented\""
        , "undefined"
        , "error $ show err"
        ]
      , suggested_fix =
          ''
          Use proper error handling:

          ```haskell
          -- BAD: Runtime exceptions
          processConfig cfg = case lookup "port" cfg of
            Nothing -> error "Port not found"  -- Don't do this
            Just p  -> startServer p

          -- GOOD: Use Either for errors
          processConfig :: Config -> Either ConfigError Server
          processConfig cfg = case lookup "port" cfg of
            Nothing -> Left (MissingField "port")
            Just p  -> Right (startServer p)

          -- GOOD: Use Maybe for optional values
          findUser :: UserId -> IO (Maybe User)
          findUser uid = queryDatabase uid

          -- GOOD: Custom exceptions for truly exceptional cases
          data AppError = ConfigurationError Text | DatabaseError Text
            deriving (Show, Exception)
          ```
          ''
      }
    , tests =
      { valid =
        [ "Left (ConfigurationError \"missing port\")"
        , "Right (startServer port)"
        , "throwIO (AppError \"failed\")"
        , "pure Nothing"
        ]
      , extra_invalid = [] : List Text
      }
    }
