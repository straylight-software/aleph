# nix/prelude/schemas.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // schemas //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'Pay no attention to those,' the Finn said. 'Got 'em in a
#      little bin behind the counter... Bought 'em off some kid, say
#      they were cellular automata. Put 'em in a jar and they do
#      that. Nothing to it, really. I don't know what they're good
#      for, but they look nice.'
#
#                                                         — Neuromancer
#
# Runtime type validation for ADTs and structured data. Use at boundaries
# to catch malformed values before they cause cryptic failures deep in
# evaluation.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   schemas.either          validate Either { _tag, value }
#   schemas.maybe           validate Maybe (null or { _tag = "Just"; value })
#   schemas.result          validate Result { ok, err }
#   schemas.validate        assert schema or throw with context
#   schemas.validate-or     return default if validation fails
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_: rec {
  # ─────────────────────────────────────────────────────────────────────────
  # Primitive Validators
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Check if a value is an attribute set.
  */
  is-attrs = builtins.isAttrs;

  /**
    Check if a value is a list.
  */
  is-list = builtins.isList;

  /**
    Check if a value is a string.
  */
  is-string = builtins.isString;

  /**
    Check if a value is an integer.
  */
  is-int = builtins.isInt;

  /**
    Check if a value is a boolean.
  */
  is-bool = builtins.isBool;

  /**
    Check if a value is a path.
  */
  is-path = builtins.isPath;

  /**
    Check if a value is a function.
  */
  is-function = builtins.isFunction;

  /**
    Check if a value is null.
  */
  is-null = x: x == null;

  # ─────────────────────────────────────────────────────────────────────────
  # ADT Validators
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Validate that a value is a well-formed Either.

    Either must be an attribute set with:
    - _tag: either "left" or "right"
    - value: the wrapped value

    # Type

    ```
    either :: a -> Bool
    ```

    # Examples

    ```nix
    either { _tag = "left"; value = "error"; }
    => true

    either { _tag = "right"; value = 42; }
    => true

    either { foo = "bar"; }
    => false

    either null
    => false

    either "not an either"
    => false
    ```
  */
  either = x: builtins.isAttrs x && x ? _tag && x ? value && (x._tag == "left" || x._tag == "right");

  /**
    Validate that a value is a well-formed Left.
  */
  left = x: either x && x._tag == "left";

  /**
    Validate that a value is a well-formed Right.
  */
  right = x: either x && x._tag == "right";

  /**
    Validate that a value is a well-formed Maybe.

    Maybe is represented as:
    - null (Nothing)
    - Any non-null value (Just)

    This is the native Nix representation, not a tagged union.

    # Type

    ```
    maybe :: a -> Bool
    ```

    # Examples

    ```nix
    maybe null
    => true

    maybe 42
    => true

    maybe { foo = "bar"; }
    => true
    ```
  */
  maybe = _x: true; # All values are valid in native Maybe representation

  /**
    Validate that a value is a well-formed Just (non-null).
  */
  just = x: x != null;

  /**
    Validate that a value is Nothing (null).
  */
  nothing = x: x == null;

  /**
    Validate that a value is a well-formed Result.

    Result must be an attribute set with exactly one of:
    - ok: the success value
    - err: the error value

    This is an alternative to Either with more explicit naming.

    # Type

    ```
    result :: a -> Bool
    ```

    # Examples

    ```nix
    result { ok = 42; }
    => true

    result { err = "something went wrong"; }
    => true

    result { ok = 1; err = 2; }
    => false

    result { }
    => false
    ```
  */
  result = x: builtins.isAttrs x && ((x ? ok && !(x ? err)) || (x ? err && !(x ? ok)));

  /**
    Validate that a value is a successful Result.
  */
  ok = x: result x && x ? ok;

  /**
    Validate that a value is an error Result.
  */
  err = x: result x && x ? err;

  # ─────────────────────────────────────────────────────────────────────────
  # List Validators
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Validate that a value is a non-empty list.

    # Type

    ```
    non-empty-list :: a -> Bool
    ```
  */
  non-empty-list = x: builtins.isList x && x != [ ];

  /**
    Validate that all elements of a list satisfy a predicate.

    # Type

    ```
    list-of :: (a -> Bool) -> [a] -> Bool
    ```

    # Examples

    ```nix
    list-of is-int [ 1 2 3 ]
    => true

    list-of is-string [ "a" "b" 3 ]
    => false
    ```
  */
  list-of = pred: xs: builtins.isList xs && builtins.all pred xs;

  # ─────────────────────────────────────────────────────────────────────────
  # Attribute Set Validators
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Validate that an attribute set has all required keys.

    # Type

    ```
    has-keys :: [String] -> AttrSet -> Bool
    ```

    # Examples

    ```nix
    has-keys [ "name" "version" ] { name = "foo"; version = "1.0"; }
    => true

    has-keys [ "name" "version" ] { name = "foo"; }
    => false
    ```
  */
  has-keys = keys: x: builtins.isAttrs x && builtins.all (k: x ? ${k}) keys;

  /**
    Validate that all values in an attribute set satisfy a predicate.

    # Type

    ```
    attrs-of :: (a -> Bool) -> AttrSet -> Bool
    ```
  */
  attrs-of = pred: x: builtins.isAttrs x && builtins.all pred (builtins.attrValues x);

  # ─────────────────────────────────────────────────────────────────────────
  # Numeric Validators
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Validate that a value is a positive integer (> 0).
  */
  positive-int = x: builtins.isInt x && x > 0;

  /**
    Validate that a value is a non-negative integer (>= 0).
  */
  non-negative-int = x: builtins.isInt x && x >= 0;

  /**
    Validate that a value is a negative integer (< 0).
  */
  negative-int = x: builtins.isInt x && x < 0;

  /**
    Validate that a value is in a range [min, max] (inclusive).

    # Type

    ```
    in-range :: Int -> Int -> Int -> Bool
    ```
  */
  in-range =
    min: max: x:
    builtins.isInt x && x >= min && x <= max;

  # ─────────────────────────────────────────────────────────────────────────
  # String Validators
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Validate that a string is non-empty.
  */
  non-empty-string = x: builtins.isString x && x != "";

  /**
    Validate that a string matches a pattern.

    # Type

    ```
    matches :: String -> String -> Bool
    ```
  */
  matches = pattern: x: builtins.isString x && builtins.match pattern x != null;

  # ─────────────────────────────────────────────────────────────────────────
  # Validation Combinators
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Combine validators with AND.

    # Type

    ```
    all-of :: [(a -> Bool)] -> a -> Bool
    ```

    # Examples

    ```nix
    all-of [ is-int positive-int ] 42
    => true

    all-of [ is-int positive-int ] (-1)
    => false
    ```
  */
  all-of = preds: x: builtins.all (p: p x) preds;

  /**
    Combine validators with OR.

    # Type

    ```
    any-of :: [(a -> Bool)] -> a -> Bool
    ```

    # Examples

    ```nix
    any-of [ is-int is-string ] 42
    => true

    any-of [ is-int is-string ] [ ]
    => false
    ```
  */
  any-of = preds: x: builtins.any (p: p x) preds;

  /**
    Negate a validator.

    # Type

    ```
    not' :: (a -> Bool) -> a -> Bool
    ```
  */
  not' = pred: x: !(pred x);

  /**
    Optional validator - null or satisfies predicate.

    # Type

    ```
    optional :: (a -> Bool) -> a -> Bool
    ```
  */
  optional = pred: x: x == null || pred x;

  # ─────────────────────────────────────────────────────────────────────────
  # Validation Actions
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Assert a value satisfies a schema, or throw with context.

    # Type

    ```
    validate :: (a -> Bool) -> String -> a -> a
    ```

    # Arguments

    - schema: predicate that returns true for valid values
    - name: name to include in error message
    - x: value to validate

    # Examples

    ```nix
    validate either "config.result" { _tag = "right"; value = 42; }
    => { _tag = "right"; value = 42; }

    validate either "config.result" { foo = "bar"; }
    => error: schema violation in 'config.result': expected Either, got { foo = "bar"; }
    ```
  */
  validate =
    schema: name: x:
    if schema x then
      x
    else
      throw "schema violation in '${name}': validation failed for ${builtins.typeOf x}";

  /**
    Validate or return a default value.

    # Type

    ```
    validate-or :: (a -> Bool) -> b -> a -> (a | b)
    ```

    # Examples

    ```nix
    validate-or positive-int 0 42
    => 42

    validate-or positive-int 0 (-1)
    => 0
    ```
  */
  validate-or =
    schema: default: x:
    if schema x then x else default;

  /**
    Validate and transform, or return a default.

    # Type

    ```
    validate-map :: (a -> Bool) -> b -> (a -> b) -> a -> b
    ```
  */
  validate-map =
    schema: default: f: x:
    if schema x then f x else default;

  # ─────────────────────────────────────────────────────────────────────────
  # Common Schema Definitions
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Schema for package metadata.
  */
  package-meta = has-keys [
    "pname"
    "version"
  ];

  /**
    Schema for derivation attributes.
  */
  derivation-attrs =
    x:
    builtins.isAttrs x
    && (x ? name || (x ? pname && x ? version))
    && (x ? src || x ? srcs || x ? unpackPhase || x ? dontUnpack);

  /**
    Schema for GPU architecture specification.
  */
  gpu-arch =
    x:
    builtins.isString x
    && builtins.elem x [
      "volta"
      "turing"
      "ampere"
      "ada"
      "hopper"
      "blackwell"
    ];

  /**
    Schema for CUDA capability version.
  */
  cuda-capability = matches "[0-9]+\\.[0-9]+";
}
