# nix/prelude/functions/attrs.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // attrs //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The Sprawl's constellations of light in the darkness, city
#     lights from the top of the arcology. The Kuang grade mark
#     eleven ice-breaker. Turing heat, black ice, the hardest
#     defenses in the matrix.
#
#                                                         — Neuromancer
#
# Attribute set operations. The fundamental data structure of Nix.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
rec {
  # ─────────────────────────────────────────────────────────────────────────
  # Attribute Sets
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Apply a function to each value in an attribute set.

    The function receives both the attribute name and value.
    Returns a new attribute set with the same keys but transformed values.

    # Type

    ```
    map-attrs :: (String -> a -> b) -> AttrSet -> AttrSet
    ```

    # Arguments

    - f: function taking name and value, returning new value
    - attrs: input attribute set

    # Examples

    ```nix
    map-attrs (name: value: value * 2) { a = 1; b = 2; }
    => { a = 2; b = 4; }

    map-attrs (name: value: "${name}=${value}") { foo = "bar"; }
    => { foo = "foo=bar"; }
    ```
  */
  map-attrs = lib.mapAttrs;

  /**
    Apply a function to each name-value pair, allowing key and value transformation.

    The function must return an attribute set with `name` and `value` fields.
    Allows renaming keys while transforming values.

    # Type

    ```
    map-attrs' :: (String -> a -> { name :: String; value :: b; }) -> AttrSet -> AttrSet
    ```

    # Arguments

    - f: function returning { name; value; } for each entry
    - attrs: input attribute set

    # Examples

    ```nix
    map-attrs' (n: v: { name = "prefix_${n}"; value = v; }) { a = 1; }
    => { prefix_a = 1; }
    ```
  */
  map-attrs' = lib.mapAttrs';

  /**
    Filter an attribute set by a predicate on name-value pairs.

    Keeps only entries where the predicate returns true.

    # Type

    ```
    filter-attrs :: (String -> a -> Bool) -> AttrSet -> AttrSet
    ```

    # Arguments

    - pred: predicate taking name and value
    - attrs: input attribute set

    # Examples

    ```nix
    filter-attrs (n: v: v > 1) { a = 1; b = 2; c = 3; }
    => { b = 2; c = 3; }

    filter-attrs (n: v: n != "secret") { user = "bob"; secret = "123"; }
    => { user = "bob"; }
    ```
  */
  filter-attrs = lib.filterAttrs;

  /**
    Left fold over an attribute set's name-value pairs.

    Reduces all entries to a single value by applying a function
    to an accumulator and each name-value pair.

    # Type

    ```
    fold-attrs :: (b -> String -> a -> b) -> b -> AttrSet -> b
    ```

    # Arguments

    - f: function taking accumulator, name, and value
    - init: initial accumulator value
    - attrs: attribute set to fold

    # Examples

    ```nix
    fold-attrs (acc: n: v: acc + v) 0 { a = 1; b = 2; c = 3; }
    => 6

    fold-attrs (acc: n: v: acc ++ [ n ]) [ ] { x = 1; y = 2; }
    => [ "x" "y" ]
    ```
  */
  fold-attrs =
    f: init: attrs:
    lib.foldlAttrs f init attrs;

  /**
    Get a sorted list of attribute names (keys) from an attribute set.

    # Type

    ```
    keys :: AttrSet -> [String]
    ```

    # Arguments

    - attrs: input attribute set

    # Examples

    ```nix
    keys { b = 2; a = 1; c = 3; }
    => [ "a" "b" "c" ]

    keys { }
    => [ ]
    ```
  */
  keys = builtins.attrNames;

  /**
    Get a list of values from an attribute set.

    Order corresponds to the sorted order of keys.

    # Type

    ```
    values :: AttrSet -> [a]
    ```

    # Arguments

    - attrs: input attribute set

    # Examples

    ```nix
    values { b = 2; a = 1; c = 3; }
    => [ 1 2 3 ]

    values { }
    => [ ]
    ```
  */
  values = builtins.attrValues;

  /**
    Check if an attribute set contains a key.

    # Type

    ```
    has :: String -> AttrSet -> Bool
    ```

    # Arguments

    - name: attribute name to check
    - attrs: attribute set to check in

    # Examples

    ```nix
    has "foo" { foo = 1; bar = 2; }
    => true

    has "baz" { foo = 1; bar = 2; }
    => false
    ```
  */
  has = builtins.hasAttr;

  /**
    Get a value from a nested attribute set by path, with a default.

    Traverses the attribute set following the path of keys.
    Returns the default if any key in the path is missing.

    # Type

    ```
    get :: [String] -> a -> AttrSet -> a
    ```

    # Arguments

    - path: list of keys to traverse
    - default: value to return if path doesn't exist
    - attrs: attribute set to query

    # Examples

    ```nix
    get [ "a" "b" ] 0 { a = { b = 42; }; }
    => 42

    get [ "a" "c" ] 0 { a = { b = 42; }; }
    => 0

    get [ ] "default" { a = 1; }
    => { a = 1; }
    ```
  */
  get = lib.attrByPath;

  /**
    Get a value from an attribute set by key (unsafe).

    Throws an error if the key doesn't exist. Use `get` for safe access.

    # Type

    ```
    get' :: String -> AttrSet -> a
    ```

    # Arguments

    - attr: attribute name to get
    - attrs: attribute set to query

    # Examples

    ```nix
    get' "foo" { foo = 42; bar = 1; }
    => 42
    ```
  */
  get' = attr: attrs: attrs.${attr};

  /**
    Create a nested attribute set from a path and value.

    # Type

    ```
    set :: [String] -> a -> AttrSet
    ```

    # Arguments

    - path: list of keys forming the path
    - value: value to place at the path

    # Examples

    ```nix
    set [ "a" "b" "c" ] 42
    => { a = { b = { c = 42; }; }; }

    set [ ] 42
    => 42

    set [ "x" ] "hello"
    => { x = "hello"; }
    ```
  */
  set = lib.setAttrByPath;

  /**
    Remove specified keys from an attribute set.

    # Type

    ```
    remove :: [String] -> AttrSet -> AttrSet
    ```

    # Arguments

    - names: list of keys to remove
    - attrs: input attribute set

    # Examples

    ```nix
    remove [ "a" "c" ] { a = 1; b = 2; c = 3; }
    => { b = 2; }

    remove [ "x" ] { a = 1; }
    => { a = 1; }
    ```
  */
  remove = names: attrs: builtins.removeAttrs attrs names;

  /**
    Update a nested attribute using a function.

    Applies a function to the value at the given path. If the path doesn't
    exist, the function receives null.

    # Type

    ```
    update :: [String] -> (a -> b) -> AttrSet -> AttrSet
    ```

    # Arguments

    - path: list of keys to the value
    - f: function to apply to the current value
    - attrs: attribute set to update

    # Examples

    ```nix
    update [ "a" ] (x: x + 1) { a = 1; b = 2; }
    => { a = 2; b = 2; }

    update [ "a" "b" ] (x: x * 2) { a = { b = 5; }; }
    => { a = { b = 10; }; }
    ```
  */
  update =
    path: f: attrs:
    lib.updateManyAttrsByPath [
      {
        inherit path;
        update = f;
      }
    ] attrs;

  /**
    Recursively merge two attribute sets (right takes precedence).

    Nested attribute sets are merged recursively. Non-attrset values
    from the right side override those from the left.

    # Type

    ```
    merge :: AttrSet -> AttrSet -> AttrSet
    ```

    # Arguments

    - left: base attribute set
    - right: attribute set to merge in (takes precedence)

    # Examples

    ```nix
    merge { a = 1; b = { x = 1; }; } { b = { y = 2; }; c = 3; }
    => { a = 1; b = { x = 1; y = 2; }; c = 3; }

    merge { a = 1; } { a = 2; }
    => { a = 2; }
    ```
  */
  merge = lib.recursiveUpdate;

  /**
    Recursively merge a list of attribute sets.

    Merges from left to right, with later sets taking precedence.

    # Type

    ```
    merge-all :: [AttrSet] -> AttrSet
    ```

    # Arguments

    - attrsList: list of attribute sets to merge

    # Examples

    ```nix
    merge-all [ { a = 1; } { b = 2; } { a = 3; c = 4; } ]
    => { a = 3; b = 2; c = 4; }

    merge-all [ ]
    => { }
    ```
  */
  merge-all = lib.foldl' lib.recursiveUpdate { };

  /**
    Convert an attribute set to a list of name-value pairs.

    Each entry becomes { name = "..."; value = ...; }.

    # Type

    ```
    to-list :: AttrSet -> [{ name :: String; value :: a; }]
    ```

    # Arguments

    - attrs: attribute set to convert

    # Examples

    ```nix
    to-list { a = 1; b = 2; }
    => [ { name = "a"; value = 1; } { name = "b"; value = 2; } ]
    ```
  */
  to-list = lib.attrsToList;

  /**
    Convert a list of name-value pairs to an attribute set.

    Each entry must have `name` and `value` fields.

    # Type

    ```
    from-list :: [{ name :: String; value :: a; }] -> AttrSet
    ```

    # Arguments

    - list: list of { name; value; } entries

    # Examples

    ```nix
    from-list [ { name = "a"; value = 1; } { name = "b"; value = 2; } ]
    => { a = 1; b = 2; }
    ```
  */
  from-list = lib.listToAttrs;

  /**
    Map over an attribute set and collect results into a list.

    Applies a function to each name-value pair and returns a list of results.

    # Type

    ```
    map-to-list :: (String -> a -> b) -> AttrSet -> [b]
    ```

    # Arguments

    - f: function taking name and value
    - attrs: attribute set to map over

    # Examples

    ```nix
    map-to-list (n: v: "${n}=${toString v}") { a = 1; b = 2; }
    => [ "a=1" "b=2" ]

    map-to-list (n: v: v * 2) { x = 1; y = 2; }
    => [ 2 4 ]
    ```
  */
  map-to-list = lib.mapAttrsToList;

  /**
    Compute the intersection of two attribute sets.

    Returns attributes from the second set that have keys present in the first.

    # Type

    ```
    intersect :: AttrSet -> AttrSet -> AttrSet
    ```

    # Arguments

    - a: attribute set defining which keys to keep
    - b: attribute set to filter (values come from here)

    # Examples

    ```nix
    intersect { a = 1; b = 2; } { a = 10; c = 30; }
    => { a = 10; }

    intersect { x = 1; } { y = 2; }
    => { }
    ```
  */
  intersect = lib.intersectAttrs;

  /**
    Generate an attribute set from a list of keys and a function.

    Creates an attribute set where each key maps to the result of
    applying the function to that key.

    # Type

    ```
    gen-attrs :: [String] -> (String -> a) -> AttrSet
    ```

    # Arguments

    - names: list of attribute names to generate
    - f: function to compute value for each name

    # Examples

    ```nix
    gen-attrs [ "a" "b" "c" ] (x: x + x)
    => { a = "aa"; b = "bb"; c = "cc"; }

    gen-attrs [ "x" "y" ] (n: 0)
    => { x = 0; y = 0; }
    ```
  */
  gen-attrs = lib.genAttrs;

  /**
    Recursively collect values matching a predicate from nested attribute sets.

    Traverses the attribute set recursively, collecting values where
    the predicate returns true.

    # Type

    ```
    collect :: (a -> Bool) -> AttrSet -> [a]
    ```

    # Arguments

    - pred: predicate to test values
    - attrs: nested attribute set to search

    # Examples

    ```nix
    collect isString { a = "x"; b = { c = "y"; d = 1; }; }
    => [ "x" "y" ]

    collect (x: x ? name) { a = { name = "foo"; }; b = { name = "bar"; }; }
    => [ { name = "foo"; } { name = "bar"; } ]
    ```
  */
  inherit (lib) collect;

  /**
    Update a value at a nested path by applying a function.

    Traverses the path and applies the function to the value at the end.
    Creates intermediate attribute sets if they don't exist.

    # Type

    ```
    update-at-path :: [String] -> (a -> b) -> AttrSet -> AttrSet
    ```

    # Arguments

    - path: list of keys to traverse
    - f: function to apply to the value at path
    - attrs: attribute set to update

    # Examples

    ```nix
    update-at-path [ "a" "b" ] (x: x + 1) { a = { b = 1; }; }
    => { a = { b = 2; }; }

    update-at-path [ "x" "y" ] (x: 42) { }
    => { x = { y = 42; }; }
    ```
  */
  update-at-path =
    path: f: attrs:
    if path == [ ] then
      f attrs
    else
      let
        head = builtins.head path;
        tail = builtins.tail path;
      in
      attrs
      // {
        ${head} = update-at-path tail f (attrs.${head} or { });
      };

  /**
    Set a value at a nested path in an attribute set.

    Creates intermediate attribute sets if they don't exist.

    # Type

    ```
    set-at-path :: [String] -> a -> AttrSet -> AttrSet
    ```

    # Arguments

    - path: list of keys to traverse
    - value: value to set at the path
    - attrs: attribute set to update

    # Examples

    ```nix
    set-at-path [ "a" "b" ] 42 { a = { c = 1; }; }
    => { a = { b = 42; c = 1; }; }

    set-at-path [ "x" "y" "z" ] "deep" { }
    => { x = { y = { z = "deep"; }; }; }
    ```
  */
  set-at-path = path: value: update-at-path path (_: value);

  /**
    Select only specified keys from an attribute set.

    Returns a new attribute set containing only the named keys.
    Missing keys are silently ignored.

    # Type

    ```
    pick :: [String] -> AttrSet -> AttrSet
    ```

    # Arguments

    - names: list of keys to keep
    - attrs: input attribute set

    # Examples

    ```nix
    pick [ "a" "c" ] { a = 1; b = 2; c = 3; }
    => { a = 1; c = 3; }

    pick [ "x" "y" ] { a = 1; }
    => { }
    ```
  */
  pick = names: attrs: lib.filterAttrs (n: _: builtins.elem n names) attrs;

  /**
    Remove specified keys from an attribute set (inverse of pick).

    Returns a new attribute set with the named keys removed.

    # Type

    ```
    omit :: [String] -> AttrSet -> AttrSet
    ```

    # Arguments

    - names: list of keys to remove
    - attrs: input attribute set

    # Examples

    ```nix
    omit [ "b" ] { a = 1; b = 2; c = 3; }
    => { a = 1; c = 3; }

    omit [ "x" ] { a = 1; b = 2; }
    => { a = 1; b = 2; }
    ```
  */
  omit = names: attrs: lib.filterAttrs (n: _: !(builtins.elem n names)) attrs;

  /**
    Rename a key in an attribute set.

    If the old key exists, it is removed and its value is assigned
    to the new key. If the old key doesn't exist, returns unchanged.

    # Type

    ```
    rename-key :: String -> String -> AttrSet -> AttrSet
    ```

    # Arguments

    - old: current key name
    - new: new key name
    - attrs: attribute set to modify

    # Examples

    ```nix
    rename-key "a" "x" { a = 1; b = 2; }
    => { x = 1; b = 2; }

    rename-key "missing" "x" { a = 1; }
    => { a = 1; }
    ```
  */
  rename-key =
    old: new: attrs:
    if builtins.hasAttr old attrs then
      builtins.removeAttrs attrs [ old ] // { ${new} = attrs.${old}; }
    else
      attrs;

  /**
    Rename multiple keys in an attribute set.

    Takes a mapping from old names to new names and applies all renames.

    # Type

    ```
    rename-keys :: AttrSet -> AttrSet -> AttrSet
    ```

    # Arguments

    - mapping: attribute set where keys are old names and values are new names
    - attrs: attribute set to transform

    # Examples

    ```nix
    rename-keys { a = "x"; b = "y"; } { a = 1; b = 2; c = 3; }
    => { x = 1; y = 2; c = 3; }

    rename-keys { old = "new"; } { old = 42; other = 1; }
    => { new = 42; other = 1; }
    ```
  */
  rename-keys =
    mapping: attrs:
    lib.foldl' (acc: old: rename-key old mapping.${old} acc) attrs (builtins.attrNames mapping);

  /**
    Invert an attribute set, swapping keys and values.

    Values must be strings (they become keys). If multiple keys
    have the same value, later ones win.

    # Type

    ```
    invert :: AttrSet -> AttrSet
    ```

    # Arguments

    - attrs: attribute set with string values

    # Examples

    ```nix
    invert { a = "x"; b = "y"; }
    => { x = "a"; y = "b"; }

    invert { foo = "bar"; baz = "bar"; }
    => { bar = "baz"; }
    ```
  */
  invert =
    attrs: lib.foldl' (acc: name: acc // { ${attrs.${name}} = name; }) { } (builtins.attrNames attrs);

  /**
    Merge default values with an attribute set.

    Values in attrs override those in defs. Useful for providing
    default configuration values.

    # Type

    ```
    defaults :: AttrSet -> AttrSet -> AttrSet
    ```

    # Arguments

    - defs: default values
    - attrs: values that override defaults

    # Examples

    ```nix
    defaults { port = 80; host = "localhost"; } { port = 8080; }
    => { port = 8080; host = "localhost"; }

    defaults { a = 1; } { b = 2; }
    => { a = 1; b = 2; }
    ```
  */
  defaults = defs: attrs: defs // attrs;

}
