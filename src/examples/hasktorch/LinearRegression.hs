{-# LANGUAGE RecordWildCards #-}

{- | Linear Regression with Hasktorch
Demonstrates basic gradient computation
-}
module Main where

import Torch

-- | Mean squared error loss
mse' :: Tensor -> Tensor -> Tensor
mse' pred target = mean $ (pred - target) * (pred - target)

-- | Generate synthetic data: y = 3x + 2 + noise
generateData :: Int -> IO (Tensor, Tensor)
generateData n = do
    x <- randnIO' [n]
    noise <- randnIO' [n]
    let y = 3.0 * x + 2.0 + 0.1 * noise
    return (x, y)

main :: IO ()
main = do
    putStrLn "═══════════════════════════════════════════════════════════"
    putStrLn "  Hasktorch - Linear Regression Demo"
    putStrLn "═══════════════════════════════════════════════════════════"
    putStrLn ""
    putStrLn "Target function: y = 3x + 2"
    putStrLn ""

    -- Generate training data
    putStrLn "Generating 100 training samples..."
    (trainX, trainY) <- generateData 100

    putStrLn $ "X shape: " ++ show (shape trainX)
    putStrLn $ "Y shape: " ++ show (shape trainY)
    putStrLn ""

    -- Initialize random weights
    putStrLn "Initializing model..."
    wInit <- randnIO' [1]
    bInit <- randnIO' [1]

    let w0 = asValue wInit :: Float
        b0 = asValue bInit :: Float
    putStrLn $ "Initial: w=" ++ show w0 ++ ", b=" ++ show b0

    -- Manual gradient descent
    putStrLn ""
    putStrLn "Training with gradient descent (500 epochs, lr=0.1)..."
    putStrLn ""

    (wFinal, bFinal) <- trainLoop wInit bInit trainX trainY 0.1 500

    let wF = asValue wFinal :: Float
        bF = asValue bFinal :: Float

    putStrLn ""
    putStrLn "═══════════════════════════════════════════════════════════"
    putStrLn $ "Learned: w=" ++ show wF ++ ", b=" ++ show bF
    putStrLn $ "Target:  w=3.0, b=2.0"
    putStrLn $
        "Error:   w_err="
            ++ show (Prelude.abs (wF - 3.0))
            ++ ", b_err="
            ++ show (Prelude.abs (bF - 2.0))
    putStrLn "═══════════════════════════════════════════════════════════"

{- | Training loop with manual gradient descent
Uses analytical gradients for MSE with linear model
-}
trainLoop ::
    Tensor ->
    Tensor ->
    Tensor ->
    Tensor ->
    Float ->
    Int ->
    IO (Tensor, Tensor)
trainLoop w b _ _ _ 0 = return (w, b)
trainLoop w b x y lr n = do
    -- Forward: pred = w*x + b
    let pred = w * x + b

        -- MSE loss
        loss = mse' pred y

        -- Analytical gradients for MSE:
        -- d(loss)/d(w) = 2 * mean((pred - y) * x)
        -- d(loss)/d(b) = 2 * mean(pred - y)
        err = pred - y
        gradW = 2.0 * mean (err * x)
        gradB = 2.0 * mean err

        -- Update
        lrT = asTensor lr
        wNew = w - lrT * gradW
        bNew = b - lrT * gradB

    -- Log every 100 epochs
    when (n `mod` 100 == 0) $ do
        let lossVal = asValue loss :: Float
            wVal = asValue w :: Float
            bVal = asValue b :: Float
        putStrLn $
            "Epoch "
                ++ show (500 - n)
                ++ ": loss="
                ++ show lossVal
                ++ ", w="
                ++ show wVal
                ++ ", b="
                ++ show bVal

    trainLoop wNew bNew x y lr (n - 1)
  where
    when True action = action
    when False _ = return ()
