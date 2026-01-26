{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

{- | Hasktorch demonstration
Shows basic tensor operations with untyped tensors (simpler API)
-}
module Main where

import Torch

-- | Simple neural network forward pass demo
main :: IO ()
main = do
    putStrLn "═══════════════════════════════════════════════════════════"
    putStrLn "  Hasktorch Demo - Tensor Operations"
    putStrLn "═══════════════════════════════════════════════════════════"
    putStrLn ""

    -- Create tensors
    putStrLn "Creating tensors..."
    let a = asTensor ([[1, 2, 3], [4, 5, 6]] :: [[Float]])
    let b = asTensor ([[7, 8], [9, 10], [11, 12]] :: [[Float]])

    putStrLn $ "Tensor a (2x3):\n" ++ show a
    putStrLn $ "\nTensor b (3x2):\n" ++ show b

    -- Matrix multiplication
    putStrLn "\n--- Matrix Multiplication ---"
    let c = matmul a b -- (2x3) @ (3x2) = (2x2)
    putStrLn $ "a @ b =\n" ++ show c

    -- Element-wise operations
    putStrLn "\n--- Element-wise Operations ---"
    let x = asTensor ([1, 2, 3, 4, 5] :: [Float])
    let y = asTensor ([5, 4, 3, 2, 1] :: [Float])

    putStrLn $ "x = " ++ show x
    putStrLn $ "y = " ++ show y
    putStrLn $ "x + y = " ++ show (x + y)
    putStrLn $ "x * y = " ++ show (x * y)
    putStrLn $ "sum(x) = " ++ show (sumAll x)

    -- Activation functions
    putStrLn "\n--- Activation Functions ---"
    let z = asTensor ([-2, -1, 0, 1, 2] :: [Float])
    putStrLn $ "z = " ++ show z
    putStrLn $ "relu(z) = " ++ show (relu z)
    putStrLn $ "sigmoid(z) = " ++ show (sigmoid z)
    putStrLn $ "tanh(z) = " ++ show (Torch.tanh z)

    -- Random tensor
    putStrLn "\n--- Random Tensors ---"
    r <- randIO' [3, 3]
    putStrLn $ "Random 3x3 tensor:\n" ++ show r

    putStrLn "\n═══════════════════════════════════════════════════════════"
    putStrLn "  Hasktorch working with aleph!"
    putStrLn "═══════════════════════════════════════════════════════════"
