{-# LANGUAGE ForeignFunctionInterface #-}

-- | FFIExport.hs - Haskell functions exported for C/C++
--
-- This module demonstrates exporting Haskell functions to C/C++
-- via the ForeignFunctionInterface extension.
--
-- When compiled with haskell_c_library, GHC generates:
--   - A static library (.a) containing the Haskell runtime and code
--   - Header files (*_stub.h) declaring the exported functions
--
-- C/C++ code can then link against this library and call the
-- exported functions, as long as it initializes the Haskell
-- runtime with hs_init() first.
module FFIExport where

import Foreign.C.Types

-- | Export: hs_double
-- Doubles an integer value
foreign export ccall hs_double :: CInt -> IO CInt

hs_double :: CInt -> IO CInt
hs_double x = return (x * 2)

-- | Export: hs_add
-- Adds two integers
foreign export ccall hs_add :: CInt -> CInt -> IO CInt

hs_add :: CInt -> CInt -> IO CInt
hs_add x y = return (x + y)

-- | Export: hs_factorial
-- Computes factorial of n
foreign export ccall hs_factorial :: CInt -> IO CInt

hs_factorial :: CInt -> IO CInt
hs_factorial n
    | n <= 1 = return 1
    | otherwise = do
        prev <- hs_factorial (n - 1)
        return (n * prev)

-- | Export: hs_fibonacci
-- Computes the nth Fibonacci number
foreign export ccall hs_fibonacci :: CInt -> IO CInt

hs_fibonacci :: CInt -> IO CInt
hs_fibonacci n
    | n <= 0 = return 0
    | n == 1 = return 1
    | otherwise = do
        a <- hs_fibonacci (n - 1)
        b <- hs_fibonacci (n - 2)
        return (a + b)

-- | Export: hs_gcd
-- Computes greatest common divisor
foreign export ccall hs_gcd :: CInt -> CInt -> IO CInt

hs_gcd :: CInt -> CInt -> IO CInt
hs_gcd a b
    | b == 0 = return (abs a)
    | otherwise = hs_gcd b (a `mod` b)
