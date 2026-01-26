{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

-- | JSON serialization demo using aeson
module Main where

import Data.Aeson
import qualified Data.ByteString.Lazy.Char8 as BL
import Data.Text (Text)
import GHC.Generics

-- | A simple person record
data Person = Person
    { name :: Text
    , age :: Int
    , email :: Text
    }
    deriving (Show, Generic)

instance ToJSON Person
instance FromJSON Person

-- | Example data
examplePerson :: Person
examplePerson = Person
    { name = "Alice"
    , age = 30
    , email = "alice@example.com"
    }

main :: IO ()
main = do
    putStrLn "═══════════════════════════════════════════════════════════"
    putStrLn "  JSON Demo - Aeson serialization"
    putStrLn "═══════════════════════════════════════════════════════════"
    putStrLn ""
    
    -- Encode to JSON
    putStrLn "Encoding Person to JSON:"
    let jsonBytes = encode examplePerson
    BL.putStrLn jsonBytes
    putStrLn ""
    
    -- Pretty print
    putStrLn "Pretty printed:"
    BL.putStrLn (encodePretty examplePerson)
    putStrLn ""
    
    -- Decode back
    putStrLn "Decoding back to Person:"
    case decode jsonBytes :: Maybe Person of
        Just p -> print p
        Nothing -> putStrLn "Failed to decode!"
    
    putStrLn ""
    putStrLn "═══════════════════════════════════════════════════════════"
    putStrLn "  aeson working with Buck2!"
    putStrLn "═══════════════════════════════════════════════════════════"

-- Pretty encoding helper (simple version)
encodePretty :: ToJSON a => a -> BL.ByteString
encodePretty = encode  -- Use regular encode, aeson-pretty would need another package
