-- | Test Haskell calling C++ via FFI.
module Main where

import Control.Monad (unless)
import FFI
import Text.Printf (printf)

main :: IO ()
main = do
    putStrLn "Haskell calling C++ via FFI:"

    -- Test arithmetic
    testArithmetic

    -- Test vector operations
    testVectors

    -- Test strings
    testStrings

    -- Test Counter
    testCounter

    putStrLn "all tests passed"

testArithmetic :: IO ()
testArithmetic = do
    let r1 = add 2 3
    check "add 2 3 == 5" (r1 == 5)

    let r2 = multiply 4 5
    check "multiply 4 5 == 20" (r2 == 20)

    putStrLn "  arithmetic: pass"

testVectors :: IO ()
testVectors = do
    let a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]

    -- dot product: 1*4 + 2*5 + 3*6 = 32
    let dp = dotProduct a b
    check "dotProduct [1,2,3] [4,5,6] == 32" (abs (dp - 32.0) < 1e-10)

    -- norm: sqrt(1 + 4 + 9) = sqrt(14)
    let n = norm a
    check "norm [1,2,3] == sqrt(14)" (abs (n - sqrt 14) < 1e-10)

    -- scale
    let scaled = scaleVector 2.0 a
    check
        "scaleVector 2 [1,2,3] == [2,4,6]"
        (all (< 1e-10) $ zipWith (\x y -> abs (x - y)) scaled [2.0, 4.0, 6.0])

    putStrLn "  vectors: pass"

testStrings :: IO ()
testStrings = do
    greeting <- greet "Haskell"
    check "greet contains 'Haskell'" ("Haskell" `isInfixOf` greeting)
    check "greet contains 'C++'" ("C++" `isInfixOf` greeting)

    putStrLn "  strings: pass"
  where
    isInfixOf needle haystack = any (needle `isPrefixOf`) (tails haystack)
    isPrefixOf [] _ = True
    isPrefixOf _ [] = False
    isPrefixOf (x : xs) (y : ys) = x == y && isPrefixOf xs ys
    tails [] = [[]]
    tails xs@(_ : xs') = xs : tails xs'

testCounter :: IO ()
testCounter = do
    withCounter 10 $ \c -> do
        v1 <- pure $ getCounter c
        check "initial value == 10" (v1 == 10)

        v2 <- incrementCounter c
        check "after increment == 11" (v2 == 11)

        v3 <- addCounter c 5
        check "after add 5 == 16" (v3 == 16)

    putStrLn "  Counter: pass"

check :: String -> Bool -> IO ()
check msg ok = unless ok $ error $ "FAILED: " ++ msg
