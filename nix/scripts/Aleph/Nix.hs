{-# LANGUAGE ForeignFunctionInterface #-}

{- | Haskell bindings for the Nix WASM host interface.

This module provides everything needed to write Nix builders in Haskell that
compile to WASM and run inside the Nix sandbox via the straylight-nix host.

= Architecture (Aleph-1 / RFC-010)

Haskell owns the control flow:

@
main = do
  -- Read the Dhall spec
  spec <- readDhallSpec "zlib-ng.dhall"
  
  -- Call Nix for fetching (FFI)
  src <- fetchGitHub (spec.src.owner) (spec.src.repo) ...
  
  -- Call Nix for dep resolution (FFI)
  cmake <- resolveDep "cmake"
  ninja <- resolveDep "ninja"
  
  -- Build
  runCMake src cmake ninja
@

The WASI host (straylight-nix) exposes:
  - Fetch primitives: fetchGitHub, fetchUrl, fetchGit
  - Store primitives: resolveDep, addToStore, getOutPath
  - Value primitives: mkInt, mkString, mkAttrs, etc.

= Modules

  * "Aleph.Nix.Fetch" - Fetch sources from GitHub, URLs, git
  * "Aleph.Nix.Store" - Resolve dependencies, add to store
  * "Aleph.Nix.Value" - Construct and inspect Nix values
  * "Aleph.Nix.Types" - Type-safe newtypes (StorePath, System, etc.)
  * "Aleph.Nix.DrvSpec" - Derivation spec types and Dhall emission
  * "Aleph.Nix.FFI" - Low-level FFI bindings

= Example

@
module MyBuilder where

import Aleph.Nix

-- The builder entry point
foreign export ccall "build" build :: IO ()

build :: IO ()
build = do
  -- Get build context from host
  out <- getOutPath "out"
  cores <- getCores
  
  -- Resolve dependencies
  cmake <- resolveDep "cmake"
  ninja <- resolveDep "ninja"
  
  -- Run the build
  -- ...
@
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
    
    -- * Fetch primitives (Aleph-1)
    fetchGitHub,
    fetchUrl,
    fetchTarball,
    fetchGit,
    
    -- * Store primitives (Aleph-1)
    resolveDep,
    resolveDeps,
    addToStore,
    getSystem,
    getCores,
    getOutPath,
) where

import Aleph.Nix.Types (NixType (..))
import Aleph.Nix.Value
import Aleph.Nix.Fetch
import Aleph.Nix.Store
