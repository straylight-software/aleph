{-# LANGUAGE ForeignFunctionInterface #-}

{- | Example Nix WASM plugin: Fibonacci numbers.

This is a minimal example showing how to write a Nix plugin in Haskell.

= Building

@
wasm32-wasi-ghc -no-hs-main -optl-mexec-model=reactor \\
  -package straylight-nix-wasm \\
  Fib.hs -o fib.wasm
@

= Usage from Nix

@
let wasm = builtins.wasm ./fib.wasm;
in {
  fib10 = wasm "fib" 10;           # => 55
  fibList = wasm "fib_list" 10;    # => [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
}
@
-}
module Aleph.Nix.Examples.Fib where

import Aleph.Nix
import Data.Int (Int64)

{- | Required initialization function.
Every WASM plugin must export this.
-}
foreign export ccall "nix_wasm_init_v1" initPlugin :: IO ()

initPlugin :: IO ()
initPlugin = nixWasmInit

{- | Compute the n-th Fibonacci number.

@wasm "fib" 10@ => 55
-}
foreign export ccall "fib" fib :: Value -> IO Value

fib :: Value -> IO Value
fib v = do
    n <- getInt v
    mkInt (fibPure n)
  where
    fibPure :: Int64 -> Int64
    fibPure 0 = 0
    fibPure 1 = 1
    fibPure n = go 0 1 (n - 1)
      where
        go a b 0 = b
        go a b k = go b (a + b) (k - 1)

{- | Return a list of the first n Fibonacci numbers.

@wasm "fib_list" 5@ => [0, 1, 1, 2, 3]
-}
foreign export ccall "fib_list" fibList :: Value -> IO Value

fibList :: Value -> IO Value
fibList v = do
    n <- getInt v
    let fibs = take (fromIntegral n) fibSequence
    values <- mapM mkInt fibs
    mkList values
  where
    fibSequence :: [Int64]
    fibSequence = 0 : 1 : zipWith (+) fibSequence (tail fibSequence)
