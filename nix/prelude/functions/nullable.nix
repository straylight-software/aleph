# nix/prelude/functions/nullable.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // nullable //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'What's he do when he's not working?'
#     'I don't know, really. Sometimes I think he's dead, or he
#      doesn't exist. Or maybe the interface just can't show me
#      what he really does.'
#
#                                                         — Neuromancer
#
# Maybe and Either. Handling absence, handling alternatives.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_: rec {
  # ─────────────────────────────────────────────────────────────────────────
  # Maybe (null handling)
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Apply a function to a nullable value, returning a default if null.

    This is the canonical way to handle nullable values in a functional style.
    If the value is null, return the default; otherwise apply the function.

    # Type

    ```
    maybe :: b -> (a -> b) -> (a | Null) -> b
    ```

    # Arguments

    - default: value to return if input is null
    - f: function to apply if input is not null
    - x: nullable input value

    # Examples

    ```nix
    maybe 0 (x: x * 2) 5
    => 10

    maybe 0 (x: x * 2) null
    => 0

    maybe "unknown" (x: "Hello, " + x) "world"
    => "Hello, world"
    ```
  */
  maybe =
    default: f: x:
    if x == null then default else f x;

  /**
    Extract a value from a nullable, using a default if null.

    Simpler version of `maybe` when no transformation is needed.

    # Type

    ```
    from-maybe :: a -> (a | Null) -> a
    ```

    # Arguments

    - default: value to return if input is null
    - x: nullable input value

    # Examples

    ```nix
    from-maybe 42 null
    => 42

    from-maybe 42 100
    => 100

    from-maybe "default" "actual"
    => "actual"
    ```
  */
  from-maybe = default: x: if x == null then default else x;

  /**
    Check if a value is null.

    # Type

    ```
    is-null :: a -> Bool
    ```

    # Arguments

    - x: value to check

    # Examples

    ```nix
    is-null null
    => true

    is-null 42
    => false

    is-null ""
    => false
    ```
  */
  is-null = x: x == null;

  /**
    Check if a value is present (not null).

    Haskell-style alias for `!is-null`. The "Just" terminology comes from
    Haskell's Maybe type where Just wraps a value.

    # Type

    ```
    is-just :: a -> Bool
    ```

    # Arguments

    - x: value to check

    # Examples

    ```nix
    is-just 42
    => true

    is-just null
    => false

    is-just ""
    => true
    ```
  */
  is-just = x: x != null;

  /**
    Check if a value is absent (null).

    Haskell-style alias for `is-null`. The "Nothing" terminology comes from
    Haskell's Maybe type.

    # Type

    ```
    is-nothing :: a -> Bool
    ```

    # Arguments

    - x: value to check

    # Examples

    ```nix
    is-nothing null
    => true

    is-nothing 42
    => false
    ```
  */
  is-nothing = x: x == null;

  /**
    Filter null values from a list, keeping only non-null elements.

    # Type

    ```
    cat-maybes :: [a | Null] -> [a]
    ```

    # Arguments

    - xs: list potentially containing null values

    # Examples

    ```nix
    cat-maybes [ 1 null 2 null 3 ]
    => [ 1 2 3 ]

    cat-maybes [ null null ]
    => [ ]

    cat-maybes [ "a" null "b" ]
    => [ "a" "b" ]
    ```
  */
  cat-maybes = builtins.filter (x: x != null);

  /**
    Map a function over a list, filtering out null results.

    Combines map and filter in one operation: applies the function to each
    element, then removes any null results from the output.

    # Type

    ```
    map-maybe :: (a -> b | Null) -> [a] -> [b]
    ```

    # Arguments

    - f: function that may return null for some inputs
    - xs: input list

    # Examples

    ```nix
    map-maybe (x: if x > 0 then x * 2 else null) [ -1 0 1 2 ]
    => [ 2 4 ]

    map-maybe (x: if x.enabled or false then x.name else null) [
      { name = "foo"; enabled = true; }
      { name = "bar"; }
      { name = "baz"; enabled = true; }
    ]
    => [ "foo" "baz" ]
    ```
  */
  map-maybe = f: xs: builtins.filter (x: x != null) (builtins.map f xs);

  # ─────────────────────────────────────────────────────────────────────────
  # Either (tagged unions)
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Construct a Left value (typically represents failure or the first alternative).

    Either is a tagged union with two variants: Left and Right. By convention,
    Left represents errors/failures and Right represents success/values.

    # Type

    ```
    left :: a -> Either a b
    ```

    # Arguments

    - value: the value to wrap in a Left

    # Examples

    ```nix
    left "error message"
    => { _tag = "left"; value = "error message"; }

    left 404
    => { _tag = "left"; value = 404; }
    ```
  */
  left = value: {
    _tag = "left";
    inherit value;
  };

  /**
    Construct a Right value (typically represents success or the second alternative).

    By convention, Right represents successful values in error handling contexts.

    # Type

    ```
    right :: b -> Either a b
    ```

    # Arguments

    - value: the value to wrap in a Right

    # Examples

    ```nix
    right 42
    => { _tag = "right"; value = 42; }

    right { name = "result"; data = [ 1 2 3 ]; }
    => { _tag = "right"; value = { name = "result"; data = [ 1 2 3 ]; }; }
    ```
  */
  right = value: {
    _tag = "right";
    inherit value;
  };

  /**
    Check if an Either value is a Left.

    Returns false for malformed values (missing _tag attribute).

    # Type

    ```
    is-left :: Either a b -> Bool
    ```

    # Arguments

    - e: an Either value

    # Examples

    ```nix
    is-left (left "error")
    => true

    is-left (right 42)
    => false

    is-left { foo = "bar"; }
    => false
    ```
  */
  is-left = e: builtins.isAttrs e && e ? _tag && e._tag == "left";

  /**
    Check if an Either value is a Right.

    Returns false for malformed values (missing _tag attribute).

    # Type

    ```
    is-right :: Either a b -> Bool
    ```

    # Arguments

    - e: an Either value

    # Examples

    ```nix
    is-right (right 42)
    => true

    is-right (left "error")
    => false

    is-right { foo = "bar"; }
    => false
    ```
  */
  is-right = e: builtins.isAttrs e && e ? _tag && e._tag == "right";

  /**
    Case analysis for Either: apply one of two functions based on the variant.

    This is the fundamental eliminator for Either values. Apply `f` if Left,
    apply `g` if Right.

    # Type

    ```
    either :: (a -> c) -> (b -> c) -> Either a b -> c
    ```

    # Arguments

    - f: function to apply if the Either is a Left
    - g: function to apply if the Either is a Right
    - e: the Either value to analyze

    # Examples

    ```nix
    either (e: "Error: " + e) (v: "Success: " + toString v) (right 42)
    => "Success: 42"

    either (e: "Error: " + e) (v: "Success: " + toString v) (left "not found")
    => "Error: not found"

    either length toString (right 123)
    => "123"
    ```
  */
  either =
    f: g: e:
    if e._tag == "left" then f e.value else g e.value;

  /**
    Extract the Right value, or return a default if Left.

    # Type

    ```
    from-right :: b -> Either a b -> b
    ```

    # Arguments

    - default: value to return if the Either is a Left
    - e: the Either value

    # Examples

    ```nix
    from-right 0 (right 42)
    => 42

    from-right 0 (left "error")
    => 0

    from-right [] (right [ 1 2 3 ])
    => [ 1 2 3 ]
    ```
  */
  from-right = default: e: if e._tag == "right" then e.value else default;

  /**
    Extract the Left value, or return a default if Right.

    # Type

    ```
    from-left :: a -> Either a b -> a
    ```

    # Arguments

    - default: value to return if the Either is a Right
    - e: the Either value

    # Examples

    ```nix
    from-left "no error" (left "something went wrong")
    => "something went wrong"

    from-left "no error" (right 42)
    => "no error"

    from-left 0 (left 404)
    => 404
    ```
  */
  from-left = default: e: if e._tag == "left" then e.value else default;

  /**
    Extract all Left values from a list of Eithers.

    # Type

    ```
    lefts :: [Either a b] -> [a]
    ```

    # Arguments

    - es: list of Either values

    # Examples

    ```nix
    lefts [ (left "a") (right 1) (left "b") (right 2) ]
    => [ "a" "b" ]

    lefts [ (right 1) (right 2) ]
    => [ ]
    ```
  */
  lefts = es: builtins.map (e: e.value) (builtins.filter (e: e._tag == "left") es);

  /**
    Extract all Right values from a list of Eithers.

    # Type

    ```
    rights :: [Either a b] -> [b]
    ```

    # Arguments

    - es: list of Either values

    # Examples

    ```nix
    rights [ (left "a") (right 1) (left "b") (right 2) ]
    => [ 1 2 ]

    rights [ (left "a") (left "b") ]
    => [ ]
    ```
  */
  rights = es: builtins.map (e: e.value) (builtins.filter (e: e._tag == "right") es);

  /**
    Partition a list of Eithers into separate lists of Lefts and Rights.

    # Type

    ```
    partition-eithers :: [Either a b] -> { lefts :: [a]; rights :: [b]; }
    ```

    # Arguments

    - es: list of Either values

    # Examples

    ```nix
    partition-eithers [ (left "a") (right 1) (left "b") (right 2) ]
    => { lefts = [ "a" "b" ]; rights = [ 1 2 ]; }

    partition-eithers [ ]
    => { lefts = [ ]; rights = [ ]; }
    ```
  */
  partition-eithers = es: {
    lefts = lefts es;
    rights = rights es;
  };

}
