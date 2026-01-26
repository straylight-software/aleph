{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}

-- | BLAKE2 hashing demo using crypton
module Main where

import Crypto.Hash (hash, Digest, SHA3_256)
import Crypto.Hash.Algorithms (Blake2b_256, Blake2b_512)
import Data.ByteString (ByteString)
import qualified Data.ByteString.Char8 as B8
import Data.ByteArray.Encoding (convertToBase, Base(Base16))

-- | Hash some data and show hex output
hashBlake256 :: ByteString -> String
hashBlake256 bs = B8.unpack $ convertToBase Base16 (hash bs :: Digest Blake2b_256)

hashBlake512 :: ByteString -> String
hashBlake512 bs = B8.unpack $ convertToBase Base16 (hash bs :: Digest Blake2b_512)

hashSHA3 :: ByteString -> String
hashSHA3 bs = B8.unpack $ convertToBase Base16 (hash bs :: Digest SHA3_256)

main :: IO ()
main = do
    putStrLn "═══════════════════════════════════════════════════════════"
    putStrLn "  BLAKE2 Hashing Demo (crypton)"
    putStrLn "═══════════════════════════════════════════════════════════"
    putStrLn ""
    
    let msg1 = "hello world" :: ByteString
    let msg2 = "The quick brown fox jumps over the lazy dog" :: ByteString
    let empty = "" :: ByteString
    
    putStrLn "BLAKE2b-256:"
    putStrLn $ "  \"\" -> " ++ hashBlake256 empty
    putStrLn $ "  \"hello world\" -> " ++ hashBlake256 msg1
    putStrLn $ "  \"The quick brown fox...\" -> " ++ hashBlake256 msg2
    putStrLn ""
    
    putStrLn "BLAKE2b-512:"
    putStrLn $ "  \"hello world\" -> " ++ hashBlake512 msg1
    putStrLn ""
    
    putStrLn "SHA3-256 (for comparison):"
    putStrLn $ "  \"hello world\" -> " ++ hashSHA3 msg1
    putStrLn ""
    
    putStrLn "═══════════════════════════════════════════════════════════"
    putStrLn "  crypton working with Buck2!"
    putStrLn "═══════════════════════════════════════════════════════════"
