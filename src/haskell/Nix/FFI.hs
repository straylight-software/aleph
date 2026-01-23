{-# LANGUAGE CApiFFI #-}
{-# LANGUAGE ForeignFunctionInterface #-}

{- | Raw FFI bindings to the Nix WASM host interface.

These are the low-level imports from the wasmtime host provided by straylight-nix.
Users should prefer the higher-level 'Aleph.Nix.Value' module.

The host provides these functions to WASM modules:
  - panic/warn for diagnostics
  - make_*/get_* for constructing and inspecting Nix values
  - copy_* for extracting compound values (strings, lists, attrsets)
  - call_function for invoking Nix functions from WASM
-}
module Aleph.Nix.FFI (
    -- * Value handle
    ValueId,

    -- * Panic and warnings
    c_panic,
    c_warn,

    -- * Type inspection
    c_get_type,

    -- * Integers
    c_make_int,
    c_get_int,

    -- * Floats
    c_make_float,
    c_get_float,

    -- * Strings
    c_make_string,
    c_copy_string,
    c_get_string_len,

    -- * Booleans
    c_make_bool,
    c_get_bool,

    -- * Null
    c_make_null,

    -- * Paths
    c_make_path,
    c_copy_path,

    -- * Lists
    c_make_list,
    c_copy_list,
    c_get_list_len,
    c_get_list_elem,

    -- * Attribute sets
    c_make_attrset,
    c_copy_attrset,
    c_copy_attrname,
    c_get_attrs_len,
    c_has_attr,
    c_get_attr,

    -- * Function calls
    c_call_function,
) where

import Data.Int (Int32, Int64)
import Data.Word (Word32, Word8)
import Foreign.C.Types (CBool (..), CDouble (..), CSize (..))
import Foreign.Ptr (Ptr)

{- | Opaque handle to a Nix value in the host.
This is an index into the host's value table.
-}
type ValueId = Word32

--------------------------------------------------------------------------------
-- Panic and warnings
--------------------------------------------------------------------------------

-- | Abort evaluation with an error message. Does not return.
foreign import ccall unsafe "panic"
    c_panic :: Ptr Word8 -> CSize -> IO ()

-- | Emit a warning message to the Nix evaluator.
foreign import ccall unsafe "warn"
    c_warn :: Ptr Word8 -> CSize -> IO ()

--------------------------------------------------------------------------------
-- Type inspection
--------------------------------------------------------------------------------

{- | Get the type tag of a value.
Returns: 1=Int, 2=Float, 3=Bool, 4=String, 5=Path, 6=Null, 7=Attrs, 8=List, 9=Function
-}
foreign import ccall unsafe "get_type"
    c_get_type :: ValueId -> IO Word32

--------------------------------------------------------------------------------
-- Integers
--------------------------------------------------------------------------------

-- | Create a Nix integer value.
foreign import ccall unsafe "make_int"
    c_make_int :: Int64 -> IO ValueId

-- | Extract the integer from a Nix value.
foreign import ccall unsafe "get_int"
    c_get_int :: ValueId -> IO Int64

--------------------------------------------------------------------------------
-- Floats
--------------------------------------------------------------------------------

-- | Create a Nix float value.
foreign import ccall unsafe "make_float"
    c_make_float :: CDouble -> IO ValueId

-- | Extract the float from a Nix value.
foreign import ccall unsafe "get_float"
    c_get_float :: ValueId -> IO CDouble

--------------------------------------------------------------------------------
-- Strings
--------------------------------------------------------------------------------

-- | Create a Nix string value from a UTF-8 buffer.
foreign import ccall unsafe "make_string"
    c_make_string :: Ptr Word8 -> CSize -> IO ValueId

{- | Copy a Nix string into a buffer.
Returns the actual length. If len > max_len, only max_len bytes are copied
but the full length is returned (so caller knows to allocate more).
-}
foreign import ccall unsafe "copy_string"
    c_copy_string :: ValueId -> Ptr Word8 -> CSize -> IO CSize

-- | Get the length of a string without copying.
foreign import ccall unsafe "get_string_len"
    c_get_string_len :: ValueId -> IO Word32

--------------------------------------------------------------------------------
-- Booleans
--------------------------------------------------------------------------------

-- | Create a Nix boolean value.
foreign import ccall unsafe "make_bool"
    c_make_bool :: CBool -> IO ValueId

-- | Extract the boolean from a Nix value.
foreign import ccall unsafe "get_bool"
    c_get_bool :: ValueId -> IO CBool

--------------------------------------------------------------------------------
-- Null
--------------------------------------------------------------------------------

-- | Create the Nix null value.
foreign import ccall unsafe "make_null"
    c_make_null :: IO ValueId

--------------------------------------------------------------------------------
-- Paths
--------------------------------------------------------------------------------

-- | Create a Nix path value from a UTF-8 buffer.
foreign import ccall unsafe "make_path"
    c_make_path :: Ptr Word8 -> CSize -> IO ValueId

{- | Copy a Nix path into a buffer.
Returns the actual length. If len > max_len, only max_len bytes are copied.
-}
foreign import ccall unsafe "copy_path"
    c_copy_path :: ValueId -> Ptr Word8 -> CSize -> IO CSize

--------------------------------------------------------------------------------
-- Lists
--------------------------------------------------------------------------------

-- | Create a Nix list from an array of value handles.
foreign import ccall unsafe "make_list"
    c_make_list :: Ptr ValueId -> CSize -> IO ValueId

{- | Copy list elements into a buffer.
Returns the actual length. If len > max_len, only max_len elements are copied.
-}
foreign import ccall unsafe "copy_list"
    c_copy_list :: ValueId -> Ptr ValueId -> CSize -> IO CSize

-- | Get the length of a list without copying.
foreign import ccall unsafe "get_list_len"
    c_get_list_len :: ValueId -> IO Word32

-- | Get a single list element by index.
foreign import ccall unsafe "get_list_elem"
    c_get_list_elem :: ValueId -> Word32 -> IO ValueId

--------------------------------------------------------------------------------
-- Attribute sets
--------------------------------------------------------------------------------

{- | Create a Nix attrset from an array of (name_ptr, name_len, value) tuples.
The layout in memory is: [ptr1, len1, val1, ptr2, len2, val2, ...]
Each tuple is 3 * sizeof(Word32) = 12 bytes (assuming 32-bit pointers in WASM32).

NOTE: The Rust version uses a different layout with &str which is (ptr, len).
We need to match what the host actually expects. Looking at wasm.cc, it seems
to expect: make_attrset(ptr, len) where ptr points to array of {ptr, len, value}.
-}
foreign import ccall unsafe "make_attrset"
    c_make_attrset :: Ptr Word8 -> CSize -> IO ValueId

{- | Copy attrset entries into a buffer.
Each entry is (ValueId, name_length) - the name is retrieved separately via copy_attrname.
Returns the number of attributes.
-}
foreign import ccall unsafe "copy_attrset"
    c_copy_attrset :: ValueId -> Ptr Word8 -> CSize -> IO CSize

{- | Copy an attribute name by index.
@copy_attrname value attr_idx buf len@ copies the name of the attr_idx'th attribute.
-}
foreign import ccall unsafe "copy_attrname"
    c_copy_attrname :: ValueId -> CSize -> Ptr Word8 -> CSize -> IO ()

-- | Get the number of attributes without copying.
foreign import ccall unsafe "get_attrs_len"
    c_get_attrs_len :: ValueId -> IO Word32

-- | Check if an attribute exists. Returns 1 if exists, 0 otherwise.
foreign import ccall unsafe "has_attr"
    c_has_attr :: ValueId -> Ptr Word8 -> CSize -> IO Int32

-- | Get an attribute by name. Returns 0xFFFFFFFF if not found.
foreign import ccall unsafe "get_attr"
    c_get_attr :: ValueId -> Ptr Word8 -> CSize -> IO Word32

--------------------------------------------------------------------------------
-- Function calls
--------------------------------------------------------------------------------

{- | Call a Nix function with arguments.
@call_function fun args_ptr args_len@ applies fun to the given arguments.
-}
foreign import ccall unsafe "call_function"
    c_call_function :: ValueId -> Ptr ValueId -> CSize -> IO ValueId
