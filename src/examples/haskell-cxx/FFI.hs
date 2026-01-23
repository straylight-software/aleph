{-# LANGUAGE ForeignFunctionInterface #-}

-- | FFI bindings to C++ code.
--
-- This module demonstrates calling C++ from Haskell via the FFI.
-- The C++ code is compiled separately and linked via Buck2's extra_libraries.
module FFI
  ( -- * Simple arithmetic
    add
  , multiply

    -- * Vector operations
  , dotProduct
  , norm
  , scaleVector

    -- * String operations
  , greet

    -- * Counter (opaque handle)
  , Counter
  , newCounter
  , freeCounter
  , withCounter
  , getCounter
  , incrementCounter
  , addCounter
  ) where

import Foreign.C.Types
import Foreign.C.String
import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.Marshal.Array
import Control.Exception (bracket)
import System.IO.Unsafe (unsafePerformIO)

-- =============================================================================
-- Simple arithmetic
-- =============================================================================

foreign import ccall unsafe "ffi_add"
  c_add :: CInt -> CInt -> CInt

foreign import ccall unsafe "ffi_multiply"
  c_multiply :: CInt -> CInt -> CInt

-- | Add two integers (calls C++).
add :: Int -> Int -> Int
add a b = fromIntegral $ c_add (fromIntegral a) (fromIntegral b)

-- | Multiply two integers (calls C++).
multiply :: Int -> Int -> Int
multiply a b = fromIntegral $ c_multiply (fromIntegral a) (fromIntegral b)

-- =============================================================================
-- Vector operations
-- =============================================================================

foreign import ccall unsafe "ffi_dot_product"
  c_dot_product :: Ptr CDouble -> Ptr CDouble -> CSize -> CDouble

foreign import ccall unsafe "ffi_norm"
  c_norm :: Ptr CDouble -> CSize -> CDouble

foreign import ccall unsafe "ffi_scale"
  c_scale :: Ptr CDouble -> CSize -> CDouble -> IO ()

-- | Compute dot product of two vectors.
dotProduct :: [Double] -> [Double] -> Double
dotProduct xs ys
  | length xs /= length ys = error "dotProduct: vectors must have same length"
  | otherwise = realToFrac $ unsafePerformIOWithArrays xs ys $ \pxs pys len ->
      return $ c_dot_product pxs pys (fromIntegral len)
  where
    unsafePerformIOWithArrays :: [Double] -> [Double] -> (Ptr CDouble -> Ptr CDouble -> Int -> IO a) -> a
    unsafePerformIOWithArrays as bs f = unsafePerformIO $
      withArray (map realToFrac as) $ \pas ->
        withArray (map realToFrac bs) $ \pbs ->
          f pas pbs (length as)



-- | Compute L2 norm of a vector.
norm :: [Double] -> Double
norm xs = unsafePerformIO $
  withArray (map realToFrac xs) $ \pxs ->
    return $ realToFrac $ c_norm pxs (fromIntegral $ length xs)

-- | Scale a vector by a scalar (returns new vector).
scaleVector :: Double -> [Double] -> [Double]
scaleVector scalar xs = unsafePerformIO $
  withArray (map realToFrac xs) $ \pxs -> do
    c_scale pxs (fromIntegral $ length xs) (realToFrac scalar)
    map realToFrac <$> peekArray (length xs) pxs

-- =============================================================================
-- String operations
-- =============================================================================

foreign import ccall unsafe "ffi_greet"
  c_greet :: CString -> IO CString

foreign import ccall unsafe "ffi_free_string"
  c_free_string :: CString -> IO ()

-- | Generate a greeting (calls C++).
greet :: String -> IO String
greet name = withCString name $ \cname -> do
  cresult <- c_greet cname
  result <- peekCString cresult
  c_free_string cresult
  return result

-- =============================================================================
-- Counter (opaque handle pattern)
-- =============================================================================

-- | Opaque handle to a C++ Counter object.
newtype Counter = Counter (Ptr Counter)

foreign import ccall unsafe "ffi_counter_new"
  c_counter_new :: CInt -> IO (Ptr Counter)

foreign import ccall unsafe "ffi_counter_free"
  c_counter_free :: Ptr Counter -> IO ()

foreign import ccall unsafe "ffi_counter_get"
  c_counter_get :: Ptr Counter -> CInt

foreign import ccall unsafe "ffi_counter_increment"
  c_counter_increment :: Ptr Counter -> IO CInt

foreign import ccall unsafe "ffi_counter_add"
  c_counter_add :: Ptr Counter -> CInt -> IO CInt

-- | Create a new counter with initial value.
newCounter :: Int -> IO Counter
newCounter initial = Counter <$> c_counter_new (fromIntegral initial)

-- | Free a counter.
freeCounter :: Counter -> IO ()
freeCounter (Counter ptr) = c_counter_free ptr

-- | Use a counter with automatic cleanup.
withCounter :: Int -> (Counter -> IO a) -> IO a
withCounter initial = bracket (newCounter initial) freeCounter

-- | Get the current value.
getCounter :: Counter -> Int
getCounter (Counter ptr) = fromIntegral $ c_counter_get ptr

-- | Increment and return new value.
incrementCounter :: Counter -> IO Int
incrementCounter (Counter ptr) = fromIntegral <$> c_counter_increment ptr

-- | Add n and return new value.
addCounter :: Counter -> Int -> IO Int
addCounter (Counter ptr) n = fromIntegral <$> c_counter_add ptr (fromIntegral n)
