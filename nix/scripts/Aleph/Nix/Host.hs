{-# LANGUAGE ForeignFunctionInterface #-}

{- | Host Contract - The interface executors must implement.

This module defines the WASI FFI contract that any executor (Nix, Buck2, or
a standalone `aleph` CLI) must implement to host Aleph builders.

= The Contract

Executors provide these capabilities via WASI imports:

== Fetching (content-addressed, cached)

@
nix_fetch_github(owner, repo, rev, hash) -> store_path
nix_fetch_url(url, hash) -> store_path
nix_fetch_git(url, rev, hash) -> store_path
@

== Store Operations

@
nix_resolve_dep(name) -> store_path
nix_add_to_store(path) -> store_path
nix_get_out_path(output_name) -> path
@

== Build Context

@
nix_get_system() -> "x86_64-linux" | "aarch64-linux" | ...
nix_get_cores() -> natural
@

== Value Marshalling (for returning results to host)

@
make_int, make_string, make_list, make_attrset, ...
get_int, get_string, get_list, get_attrset, ...
@

= Executor Implementations

== Nix (straylight-nix)

The Nix executor implements these as calls into the Nix evaluator:
- fetch_github -> builtins.fetchFromGitHub
- resolve_dep -> pkgs.${name} lookup
- get_out_path -> placeholder "out"

== Buck2

The Buck2 executor implements these as:
- fetch_github -> buck2 fetch action
- resolve_dep -> target resolution from BUCK files
- get_out_path -> $(location :output)

== Standalone CLI

A standalone `aleph` CLI could implement these as:
- fetch_github -> git clone + cache
- resolve_dep -> local store lookup
- get_out_path -> temp directory

= Why WASI?

WASI provides:
1. Sandboxing - builders can't do arbitrary I/O
2. Portability - same .wasm runs on Nix, Buck2, or standalone
3. Language agnostic - builders can be Haskell, Rust, Zig, etc.

The builder is a pure function: Spec -> Actions
The executor provides: fetch, resolve, store
-}
module Aleph.Nix.Host (
    -- * Host capability check
    hostCapabilities,
    HostCapabilities (..),
    
    -- * Re-export the contract
    module Aleph.Nix.Fetch,
    module Aleph.Nix.Store,
) where

import Aleph.Nix.Fetch
import Aleph.Nix.Store

import Data.Text (Text)

-- | Capabilities provided by the host executor.
data HostCapabilities = HostCapabilities
    { canFetchGitHub :: Bool
    , canFetchUrl :: Bool
    , canFetchGit :: Bool
    , canResolveDep :: Bool
    , canAddToStore :: Bool
    , hasValueFFI :: Bool  -- Can marshal Nix values (only Nix executor)
    }
    deriving (Show)

-- | Query host capabilities.
-- This allows builders to adapt to different executors.
hostCapabilities :: IO HostCapabilities
hostCapabilities = do
    -- TODO: Actually query the host via FFI
    -- For now, assume full Nix capabilities
    pure HostCapabilities
        { canFetchGitHub = True
        , canFetchUrl = True
        , canFetchGit = True
        , canResolveDep = True
        , canAddToStore = True
        , hasValueFFI = True
        }
