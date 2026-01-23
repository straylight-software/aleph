{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE OverloadedStrings #-}

{- | Example Nix WASM plugin: Derivation helpers.

This example shows how to work with attrsets and demonstrates
a potential use case: building derivation specifications in Haskell.

= Usage from Nix

@
let wasm = builtins.wasm ./derivation.wasm;
in {
  # Normalize a derivation spec (add defaults, validate)
  myDrv = wasm "normalize_drv" {
    name = "hello";
    src = ./src;
  };

  # Merge two attrsets (like // but from WASM)
  merged = wasm "merge" { a = 1; } { b = 2; };
}
@
-}
module Aleph.Nix.Examples.Derivation where

import Aleph.Nix
import Data.Int (Int64)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Text (Text)
import qualified Data.Text as T

-- | Required initialization function.
foreign export ccall "nix_wasm_init_v1" initPlugin :: IO ()

initPlugin :: IO ()
initPlugin = nixWasmInit

{- | Normalize a derivation specification by adding defaults.

Input: { name, src, ... }
Output: { name, src, builder, system, ... } with defaults filled in
-}
foreign export ccall "normalize_drv" normalizeDrv :: Value -> IO Value

normalizeDrv :: Value -> IO Value
normalizeDrv v = do
    attrs <- getAttrs v

    -- Check required fields
    case Map.lookup "name" attrs of
        Nothing -> panic "normalize_drv: missing required field 'name'"
        Just _ -> pure ()

    -- Add defaults for optional fields
    defaults <-
        sequence $
            Map.fromList
                [ ("system", mkString "x86_64-linux")
                , ("builder", mkString "/bin/sh")
                ]

    -- Merge: defaults ++ input (input wins)
    let merged = Map.union attrs defaults
    mkAttrs merged

{- | Merge two attrsets, with the second taking precedence.

This is equivalent to Nix's @a // b@.
-}
foreign export ccall "merge" merge :: Value -> Value -> IO Value

merge :: Value -> Value -> IO Value
merge va vb = do
    a <- getAttrs va
    b <- getAttrs vb
    mkAttrs (Map.union b a) -- union prefers first arg on conflict

{- | Get the type of a value as a string.

Useful for debugging and type introspection.
-}
foreign export ccall "type_of" typeOf :: Value -> IO Value

typeOf :: Value -> IO Value
typeOf v = do
    t <- getType v
    mkString $ case t of
        NixInt -> "int"
        NixFloat -> "float"
        NixBool -> "bool"
        NixString -> "string"
        NixPath -> "path"
        NixNull -> "null"
        NixAttrs -> "set"
        NixList -> "list"
        NixFunction -> "lambda"
        NixUnknown n -> "unknown:" <> T.pack (show n)

-- | Count the number of attributes in an attrset.
foreign export ccall "attr_count" attrCount :: Value -> IO Value

attrCount :: Value -> IO Value
attrCount v = do
    attrs <- getAttrs v
    mkInt (fromIntegral $ Map.size attrs)

-- | Get the keys of an attrset as a list of strings.
foreign export ccall "attr_names" attrNames :: Value -> IO Value

attrNames :: Value -> IO Value
attrNames v = do
    attrs <- getAttrs v
    names <- mapM mkString (Map.keys attrs)
    mkList names
