{-# LANGUAGE ForeignFunctionInterface #-}
module Main where

import Aleph.Nix.Value (Value(..))
import Aleph.Nix.Derivation (drvToNixAttrs)
import Aleph.Nix (nixWasmInit)
import qualified Pkg (pkg)

main :: IO ()
main = pure ()

foreign export ccall "nix_wasm_init_v1" initPlugin :: IO ()
initPlugin :: IO ()
initPlugin = nixWasmInit

foreign export ccall "pkg" pkgExport :: Value -> IO Value
pkgExport :: Value -> IO Value
pkgExport _args = drvToNixAttrs Pkg.pkg
