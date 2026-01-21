{-# LANGUAGE LambdaCase #-}

-- | Aleph.Build.Elf
-- ELF manipulation utilities
module Aleph.Build.Elf
  ( patchRpath
  , patchInterpreter
  , shrinkRpath
  , readRpath
  , readInterpreter
  , isElfFile
  ) where

import Data.List (intercalate)
import System.Process (callProcess, readProcess)
import qualified Data.ByteString as BS

-- | Set rpath on an ELF binary
patchRpath :: FilePath -> [FilePath] -> IO ()
patchRpath binary rpaths = do
  let rpathStr = intercalate ":" rpaths
  callProcess "patchelf" ["--set-rpath", rpathStr, binary]

-- | Set interpreter on an ELF binary
patchInterpreter :: FilePath -> FilePath -> IO ()
patchInterpreter binary interpreter =
  callProcess "patchelf" ["--set-interpreter", interpreter, binary]

-- | Remove unused rpath entries
shrinkRpath :: FilePath -> IO ()
shrinkRpath binary =
  callProcess "patchelf" ["--shrink-rpath", binary]

-- | Read current rpath
readRpath :: FilePath -> IO [FilePath]
readRpath binary = do
  out <- readProcess "patchelf" ["--print-rpath", binary] ""
  pure $ filter (not . null) $ splitOn ':' $ strip out
  where
    strip = reverse . dropWhile (== '\n') . reverse
    splitOn _ "" = []
    splitOn c s = case break (== c) s of
      (x, "") -> [x]
      (x, _ : rest) -> x : splitOn c rest

-- | Read current interpreter
readInterpreter :: FilePath -> IO FilePath
readInterpreter binary = do
  out <- readProcess "patchelf" ["--print-interpreter", binary] ""
  pure $ strip out
  where
    strip = reverse . dropWhile (== '\n') . reverse

-- | Check if file is an ELF binary
isElfFile :: FilePath -> IO Bool
isElfFile path = do
  contents <- BS.readFile path
  pure $ BS.take 4 contents == elfMagic
  where
    elfMagic = BS.pack [0x7f, 0x45, 0x4c, 0x46]  -- \x7fELF
