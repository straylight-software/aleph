#!/usr/bin/env runhaskell
{-# LANGUAGE OverloadedStrings #-}

-- Test CMake module against known-good flags

import Aleph.Script.Tools.CMake
import Control.Monad (unless)
import Data.Text (Text)
import qualified Data.Text as T

main :: IO ()
main = do
    putStrLn "Testing CMake.buildArgs..."

    putStrLn "\n=== zlib-ng test ==="
    let zlibResult = testZlibNgFlags
    putStrLn $ "Pass: " ++ show zlibResult
    unless zlibResult $ do
        putStrLn "Expected:"
        mapM_ (putStrLn . ("  " ++) . T.unpack) expectedZlibNg
        putStrLn "Got:"
        mapM_ (putStrLn . ("  " ++) . T.unpack) actualZlibNg

    putStrLn "\n=== fmt test ==="
    let fmtResult = testFmtFlags
    putStrLn $ "Pass: " ++ show fmtResult
    unless fmtResult $ do
        putStrLn "Expected:"
        mapM_ (putStrLn . ("  " ++) . T.unpack) expectedFmt
        putStrLn "Got:"
        mapM_ (putStrLn . ("  " ++) . T.unpack) actualFmt

    putStrLn "\n=== Summary ==="
    if zlibResult && fmtResult
        then putStrLn "All tests passed!"
        else putStrLn "SOME TESTS FAILED"

-- Helpers to show what we actually produced
expectedZlibNg, actualZlibNg :: [Text]
expectedZlibNg =
    [ "-DCMAKE_INSTALL_PREFIX=/"
    , "-DBUILD_SHARED_LIBS=OFF"
    , "-DBUILD_STATIC_LIBS=ON"
    , "-DINSTALL_UTILS=ON"
    , "-DZLIB_COMPAT=ON"
    ]
actualZlibNg =
    buildArgs
        defaults
            { installPrefix = Just "/"
            , buildStaticLibs = Just True
            , buildSharedLibs = Just False
            , extraFlags =
                [ ("INSTALL_UTILS", "ON")
                , ("ZLIB_COMPAT", "ON")
                ]
            }

expectedFmt, actualFmt :: [Text]
expectedFmt =
    [ "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    , "-DBUILD_SHARED_LIBS=OFF"
    , "-DBUILD_STATIC_LIBS=ON"
    , "-DCMAKE_CXX_STANDARD=17"
    , "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    ]
actualFmt =
    buildArgs
        defaults
            { buildType = Just RelWithDebInfo
            , cxxStandard = Just 17
            , positionIndependentCode = Just True
            , buildStaticLibs = Just True
            , buildSharedLibs = Just False
            }
