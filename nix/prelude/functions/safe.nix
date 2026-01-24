# nix/prelude/functions/safe.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // safe //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     Case heard Molly stir beside him. 'Linda,' she said. 'Who's
#      Linda?' Her voice, throaty, sleep-blurred.
#
#     And he answered, his own voice almost alien in his ears,
#     nothing at all.
#
#                                                         — Neuromancer
#
# Total functions that never crash. Every function here:
#   - Accepts all inputs in its domain (including edge cases)
#   - Returns Maybe (null | value) or Either for partial operations
#   - Has a corresponding -unsafe variant for those who know better
#
# Principle: Safe by default, explicitly unsafe when needed.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   safe.head         null on empty list (vs crash)
#   safe.tail         [] on empty list (vs crash)
#   safe.minimum      null on empty list (vs crash)
#   safe.maximum      null on empty list (vs crash)
#   safe.elem-at      null on out-of-bounds (vs crash)
#   safe.take         clamp negative to 0 (vs crash)
#   safe.drop         clamp negative to 0 (vs crash)
#   safe.chunks-of    [] for n <= 0 (vs infinite loop)
#   safe.div          null for division by zero (vs crash)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
rec {
  # ─────────────────────────────────────────────────────────────────────────
  # List Operations
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Safely get the first element of a list.

    Returns null for empty lists instead of crashing.

    # Type

    ```
    head :: [a] -> (a | Null)
    ```

    # Examples

    ```nix
    head [ 1 2 3 ]
    => 1

    head [ ]
    => null
    ```
  */
  head = xs: if xs == [ ] then null else builtins.head xs;

  /**
    Safely get the last element of a list.

    Returns null for empty lists instead of crashing.

    # Type

    ```
    last :: [a] -> (a | Null)
    ```

    # Examples

    ```nix
    last [ 1 2 3 ]
    => 3

    last [ ]
    => null
    ```
  */
  last = xs: if xs == [ ] then null else lib.last xs;

  /**
    Safely get all but the first element of a list.

    Returns empty list for empty/singleton lists instead of crashing.

    # Type

    ```
    tail :: [a] -> [a]
    ```

    # Examples

    ```nix
    tail [ 1 2 3 ]
    => [ 2 3 ]

    tail [ 1 ]
    => [ ]

    tail [ ]
    => [ ]
    ```
  */
  tail = xs: if builtins.length xs <= 1 then [ ] else builtins.tail xs;

  /**
    Safely get all but the last element of a list.

    Returns empty list for empty/singleton lists instead of crashing.

    # Type

    ```
    init :: [a] -> [a]
    ```

    # Examples

    ```nix
    init [ 1 2 3 ]
    => [ 1 2 ]

    init [ 1 ]
    => [ ]

    init [ ]
    => [ ]
    ```
  */
  init = xs: if builtins.length xs <= 1 then [ ] else lib.init xs;

  /**
    Safely get element at index.

    Returns null for out-of-bounds indices instead of crashing.

    # Type

    ```
    elem-at :: [a] -> Int -> (a | Null)
    ```

    # Examples

    ```nix
    elem-at [ "a" "b" "c" ] 1
    => "b"

    elem-at [ "a" "b" "c" ] 10
    => null

    elem-at [ "a" "b" "c" ] (-1)
    => null
    ```
  */
  elem-at = xs: i: if i < 0 || i >= builtins.length xs then null else builtins.elemAt xs i;

  /**
    Safely take n elements from a list.

    Clamps negative n to 0, never crashes.

    # Type

    ```
    take :: Int -> [a] -> [a]
    ```

    # Examples

    ```nix
    take 2 [ 1 2 3 4 ]
    => [ 1 2 ]

    take (-5) [ 1 2 3 ]
    => [ ]

    take 100 [ 1 2 ]
    => [ 1 2 ]
    ```
  */
  take = n: xs: lib.take (if n < 0 then 0 else n) xs;

  /**
    Safely drop n elements from a list.

    Clamps negative n to 0, never crashes.

    # Type

    ```
    drop :: Int -> [a] -> [a]
    ```

    # Examples

    ```nix
    drop 2 [ 1 2 3 4 ]
    => [ 3 4 ]

    drop (-5) [ 1 2 3 ]
    => [ 1 2 3 ]

    drop 100 [ 1 2 ]
    => [ ]
    ```
  */
  drop = n: xs: lib.drop (if n < 0 then 0 else n) xs;

  /**
    Safely split a list into chunks of size n.

    Returns empty list for n <= 0, preventing infinite loops.

    # Type

    ```
    chunks-of :: Int -> [a] -> [[a]]
    ```

    # Examples

    ```nix
    chunks-of 2 [ 1 2 3 4 5 ]
    => [ [ 1 2 ] [ 3 4 ] [ 5 ] ]

    chunks-of 0 [ 1 2 3 ]
    => [ ]

    chunks-of (-1) [ 1 2 3 ]
    => [ ]
    ```
  */
  chunks-of =
    n: xs:
    if n <= 0 then
      [ ]
    else if xs == [ ] then
      [ ]
    else
      [ (lib.take n xs) ] ++ chunks-of n (lib.drop n xs);

  /**
    Safely find the minimum element.

    Returns null for empty lists instead of crashing.

    # Type

    ```
    minimum :: [Ord] -> (Ord | Null)
    ```

    # Examples

    ```nix
    minimum [ 3 1 4 1 5 ]
    => 1

    minimum [ ]
    => null
    ```
  */
  minimum =
    xs:
    if xs == [ ] then
      null
    else
      lib.foldl' (a: b: if a < b then a else b) (builtins.head xs) (builtins.tail xs);

  /**
    Safely find the maximum element.

    Returns null for empty lists instead of crashing.

    # Type

    ```
    maximum :: [Ord] -> (Ord | Null)
    ```

    # Examples

    ```nix
    maximum [ 3 1 4 1 5 ]
    => 5

    maximum [ ]
    => null
    ```
  */
  maximum =
    xs:
    if xs == [ ] then
      null
    else
      lib.foldl' (a: b: if a > b then a else b) (builtins.head xs) (builtins.tail xs);

  /**
    Safely find the minimum element by a key function.

    Returns null for empty lists instead of crashing.

    # Type

    ```
    minimum-by :: (a -> Ord) -> [a] -> (a | Null)
    ```
  */
  minimum-by =
    f: xs:
    if xs == [ ] then
      null
    else
      lib.foldl' (a: b: if (f a) < (f b) then a else b) (builtins.head xs) (builtins.tail xs);

  /**
    Safely find the maximum element by a key function.

    Returns null for empty lists instead of crashing.

    # Type

    ```
    maximum-by :: (a -> Ord) -> [a] -> (a | Null)
    ```
  */
  maximum-by =
    f: xs:
    if xs == [ ] then
      null
    else
      lib.foldl' (a: b: if (f a) > (f b) then a else b) (builtins.head xs) (builtins.tail xs);

  /**
    Safely find the first element matching a predicate.

    Returns null if no element matches.

    # Type

    ```
    find :: (a -> Bool) -> [a] -> (a | Null)
    ```

    # Examples

    ```nix
    find (x: x > 3) [ 1 2 3 4 5 ]
    => 4

    find (x: x > 10) [ 1 2 3 ]
    => null
    ```
  */
  find = lib.findFirst;

  # ─────────────────────────────────────────────────────────────────────────
  # Arithmetic Operations
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Safely divide two numbers.

    Returns null for division by zero instead of crashing.

    # Type

    ```
    div :: Num -> Num -> (Num | Null)
    ```

    # Examples

    ```nix
    div 10 2
    => 5

    div 10 0
    => null
    ```
  */
  div = a: b: if b == 0 then null else a / b;

  /**
    Safely compute modulo.

    Returns null for modulo by zero instead of crashing.

    # Type

    ```
    mod :: Int -> Int -> (Int | Null)
    ```
  */
  mod = a: b: if b == 0 then null else lib.mod a b;

  # ─────────────────────────────────────────────────────────────────────────
  # String Operations
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Safely get a substring.

    Clamps indices to valid range, never crashes.

    # Type

    ```
    substring :: Int -> Int -> String -> String
    ```

    # Examples

    ```nix
    substring 0 3 "hello"
    => "hel"

    substring (-5) 3 "hello"
    => "hel"

    substring 0 100 "hi"
    => "hi"
    ```
  */
  substring =
    start: len: s:
    let
      slen = builtins.stringLength s;
      start' =
        if start < 0 then
          0
        else if start > slen then
          slen
        else
          start;
      len' =
        if len < 0 then
          0
        else if start' + len > slen then
          slen - start'
        else
          len;
    in
    builtins.substring start' len' s;

  /**
    Safely get a character at index.

    Returns null for out-of-bounds indices.

    # Type

    ```
    char-at :: Int -> String -> (String | Null)
    ```
  */
  char-at = i: s: if i < 0 || i >= builtins.stringLength s then null else builtins.substring i 1 s;

  # ─────────────────────────────────────────────────────────────────────────
  # Attribute Set Operations
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Safely get an attribute from a set.

    Returns null if attribute doesn't exist.

    This is the canonical Maybe-style safe accessor for attribute sets,
    analogous to Haskell's `Data.Map.lookup :: k -> Map k v -> Maybe v`.
    The `or null` here is the *implementation* of the Maybe pattern, not
    a defensive hack — other code should use this function instead of
    inline `or null` patterns.

    NOTE: Exempt from or-null-fallback lint rule — this IS the safe accessor.

    # Type

    ```
    get-attr :: String -> AttrSet -> (a | Null)
    ```

    # Examples

    ```nix
    get-attr "foo" { foo = 42; bar = "hi"; }
    => 42

    get-attr "baz" { foo = 42; }
    => null
    ```
  */
  get-attr = name: attrs: attrs.${name} or null;

  /**
    Safely get a nested attribute using a path.

    Returns null if any part of the path doesn't exist.

    This is the canonical Maybe-style safe path traversal, analogous to
    lens-style safe access (like `preview` in Haskell's lens library).
    It correctly short-circuits on null and uses `or null` at each step
    to implement the Maybe pattern — other code should use this function
    instead of inline `or null` patterns for nested access.

    NOTE: Exempt from or-null-fallback lint rule — this IS the safe accessor.

    # Type

    ```
    get-path :: [String] -> AttrSet -> (a | Null)
    ```

    # Examples

    ```nix
    get-path [ "a" "b" "c" ] { a = { b = { c = 42; }; }; }
    => 42

    get-path [ "a" "x" "c" ] { a = { b = { c = 42; }; }; }
    => null
    ```
  */
  get-path =
    path: attrs:
    builtins.foldl' (acc: key: if acc == null then null else acc.${key} or null) attrs path;

  # ─────────────────────────────────────────────────────────────────────────
  # Either Operations (safe by nature, but included for completeness)
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Check if a value is a well-formed Either before operating.

    Returns null if the value is not an Either.

    # Type

    ```
    is-left :: a -> (Bool | Null)
    ```
  */
  is-left = e: if builtins.isAttrs e && e ? _tag && e ? value then e._tag == "left" else null;

  /**
    Check if a value is a well-formed Right.

    Returns null if the value is not an Either.

    # Type

    ```
    is-right :: a -> (Bool | Null)
    ```
  */
  is-right = e: if builtins.isAttrs e && e ? _tag && e ? value then e._tag == "right" else null;

  # ─────────────────────────────────────────────────────────────────────────
  # Conversion to Result Type
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Convert a nullable value to a Result.

    # Type

    ```
    to-result :: String -> (a | Null) -> Result String a
    ```

    # Examples

    ```nix
    to-result "not found" (head [ 1 2 3 ])
    => { ok = 1; }

    to-result "empty list" (head [ ])
    => { err = "empty list"; }
    ```
  */
  to-result = errMsg: x: if x == null then { err = errMsg; } else { ok = x; };

  /**
    Try an operation, catching errors as Left.

    Note: This uses builtins.tryEval which only catches certain errors.

    # Type

    ```
    try :: a -> Either String a
    ```
  */
  try =
    x:
    let
      result = builtins.tryEval x;
    in
    if result.success then
      {
        _tag = "right";
        inherit (result) value;
      }
    else
      {
        _tag = "left";
        value = "evaluation failed";
      };
}
