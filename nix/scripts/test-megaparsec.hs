{-# LANGUAGE OverloadedStrings #-}

-- Quick test for megaparsec CLI option parsing

import Data.Text (Text)
import Data.Void
import Text.Megaparsec
import Text.Megaparsec.Char

type Parser = Parsec Void Text

-- Simple clap-style option parser
clapOption :: Parser (Text, Text)
clapOption = do
    _ <- string "--"
    name <- takeWhile1P (Just "option name") (/= ' ')
    _ <- space
    _ <- char '<'
    typ <- takeWhile1P (Just "type") (/= '>')
    _ <- char '>'
    return (name, typ)

main :: IO ()
main = case parse clapOption "" "--output <FILE>" of
    Left e -> putStrLn $ errorBundlePretty e
    Right (n, t) -> putStrLn $ "Parsed: " ++ show n ++ " : " ++ show t
