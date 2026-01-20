{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE OverloadedStrings #-}

{- | Single-file package support for call-package.

This module re-exports everything needed to write a typed package
definition in a single .hs file.

= Minimal Example

@
{\-# LANGUAGE ForeignFunctionInterface #-\}
{\-# LANGUAGE OverloadedStrings #-\}
module Main where

import Aleph.Nix.Package

-- The required exports (copy this block verbatim)
main :: IO ()
main = pure ()

foreign export ccall "nix_wasm_init_v1" initPlugin :: IO ()
initPlugin :: IO ()
initPlugin = nixWasmInit

foreign export ccall "pkg" pkgExport :: Value -> IO Value
pkgExport :: Value -> IO Value
pkgExport _args = drvToNixAttrs pkg

-- Your package definition
pkg :: Drv
pkg = mkDerivation
    [ pname "my-package"
    , version "1.0.0"
    , src $ fetchFromGitHub
        [ owner "org"
        , repo "my-package"
        , rev "v1.0.0"
        , hash "sha256-..."
        ]
    , cmake defaults  -- or: cmakeFlags [ "-DBUILD_TESTING=OFF" ]
    , description "My typed package"
    , license "mit"
    ]
@

Usage from Nix:

@
  my-package = call-package ./my-package.hs {};
@
-}
module Aleph.Nix.Package (
    -- * Core types
    Drv,
    Value (..),

    -- * Package definition
    mkDerivation,

    -- * Basic attributes
    pname,
    version,
    src,
    strictDeps,
    doCheck,
    dontUnpack,

    -- * Source fetching
    fetchFromGitHub,
    fetchurl,
    owner,
    repo,
    rev,
    hash,
    url,

    -- * Dependencies
    nativeBuildInputs,
    buildInputs,
    propagatedBuildInputs,
    checkInputs,
    patches,

    -- * Build systems
    cmake,
    cmakeFlags,
    configureFlags,
    makeFlags,
    mesonFlags,

    -- * CMake options (typed)
    defaults,
    Options (..),
    BuildType (..),

    -- * Phases (typed actions)
    postPatch,
    preConfigure,
    installPhase,
    postInstall,
    postFixup,

    -- * Phase actions
    writeFile,
    substitute,
    mkdir,
    copy,
    symlink,
    remove,
    run,
    tool,
    patchElfRpath,
    patchElfAddRpath,
    wrap,
    unzip,

    -- * Wrap actions
    wrapPrefix,
    wrapSuffix,
    wrapSet,
    wrapSetDefault,
    wrapUnset,
    wrapAddFlags,

    -- * Metadata
    description,
    homepage,
    license,
    mainProgram,

    -- * Marshaling
    drvToNixAttrs,

    -- * Initialization
    nixWasmInit,
) where

import Prelude hiding (unzip, writeFile)

import Aleph.Nix (nixWasmInit)
import Aleph.Nix.Derivation (Drv, drvToNixAttrs)
import Aleph.Nix.Syntax
import Aleph.Nix.Value (Value (..))
