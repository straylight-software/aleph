{-# LANGUAGE ForeignFunctionInterface #-}

{- | Haskell bindings for the Nix WASM host interface.

This module provides everything needed to write Nix plugins in Haskell that
compile to WASM and run inside Nix evaluation via @builtins.wasm@.

= Quick Start

1. Write your plugin functions:

@
module MyPlugin where

import Aleph.Nix

-- Export the init function (required)
foreign export ccall "nix_wasm_init_v1" initPlugin :: IO ()
initPlugin :: IO ()
initPlugin = nixWasmInit

-- Export your functions
foreign export ccall "double" doubleInt :: Value -> IO Value
doubleInt :: Value -> IO Value
doubleInt v = do
  n <- getInt v
  mkInt (n * 2)
@

2. Compile to WASM:

@
wasm32-wasi-ghc -no-hs-main -optl-mexec-model=reactor MyPlugin.hs -o my_plugin.wasm
@

3. Use from Nix:

@
let wasm = builtins.wasm ./my_plugin.wasm;
in wasm "double" 21  # => 42
@

= Architecture

WASM modules run in a sandboxed environment provided by wasmtime inside the
Nix evaluator (straylight-nix). The host provides functions for:

* Creating and inspecting Nix values (integers, strings, lists, attrsets, etc.)
* Calling Nix functions from WASM
* Emitting warnings and aborting evaluation

Values are represented as opaque 32-bit handles. The actual values live in
the host's memory and are garbage-collected normally by Nix.
-}
module Aleph.Nix (
    -- * Value type
    Value (..),
    NixType (..),
    getType,

    -- * Constructing values
    mkInt,
    mkFloat,
    mkString,
    mkBool,
    mkNull,
    mkList,
    mkAttrs,

    -- * Inspecting values
    getInt,
    getFloat,
    getString,
    getBool,
    getList,
    getAttrs,

    -- * Calling Nix functions
    call,
    call1,
    call2,

    -- * Diagnostics
    panic,
    warn,

    -- * Module initialization
    nixWasmInit,
) where

import Aleph.Nix.Types (NixType (..))
import Aleph.Nix.Value
