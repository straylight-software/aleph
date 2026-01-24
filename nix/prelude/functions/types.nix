# nix/prelude/functions/types.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // types //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'Minds aren't read. See, you've still got the paradigms print
#      gave you, and you're barely print-Loss. I can access your
#      memory, but that's not the same as your mind.'
#
#                                                         — Neuromancer
#
# Type predicates. is-list, is-string, is-attrs — knowing what you have.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_: {
  # ─────────────────────────────────────────────────────────────────────────
  # Type predicates
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Check if a value is a list.

    # Type

    ```
    is-list :: a -> Bool
    ```

    # Arguments

    - x: value to check

    # Examples

    ```nix
    is-list [ 1 2 3 ]
    => true

    is-list "not a list"
    => false

    is-list { }
    => false
    ```
  */
  is-list = builtins.isList;

  /**
    Check if a value is an attribute set.

    # Type

    ```
    is-attrs :: a -> Bool
    ```

    # Arguments

    - x: value to check

    # Examples

    ```nix
    is-attrs { a = 1; }
    => true

    is-attrs [ 1 2 ]
    => false

    is-attrs null
    => false
    ```
  */
  is-attrs = builtins.isAttrs;

  /**
    Check if a value is a string.

    # Type

    ```
    is-string :: a -> Bool
    ```

    # Arguments

    - x: value to check

    # Examples

    ```nix
    is-string "hello"
    => true

    is-string 42
    => false

    is-string [ "a" ]
    => false
    ```
  */
  is-string = builtins.isString;

  /**
    Check if a value is an integer.

    # Type

    ```
    is-int :: a -> Bool
    ```

    # Arguments

    - x: value to check

    # Examples

    ```nix
    is-int 42
    => true

    is-int 3.14
    => false

    is-int "42"
    => false
    ```
  */
  is-int = builtins.isInt;

  /**
    Check if a value is a boolean.

    # Type

    ```
    is-bool :: a -> Bool
    ```

    # Arguments

    - x: value to check

    # Examples

    ```nix
    is-bool true
    => true

    is-bool false
    => true

    is-bool 1
    => false
    ```
  */
  is-bool = builtins.isBool;

  /**
    Check if a value is a floating-point number.

    # Type

    ```
    is-float :: a -> Bool
    ```

    # Arguments

    - x: value to check

    # Examples

    ```nix
    is-float 3.14
    => true

    is-float 42
    => false

    is-float "3.14"
    => false
    ```
  */
  is-float = builtins.isFloat;

  /**
    Check if a value is a path.

    # Type

    ```
    is-path :: a -> Bool
    ```

    # Arguments

    - x: value to check

    # Examples

    ```nix
    is-path ./foo
    => true

    is-path "/foo"
    => false

    is-path { }
    => false
    ```
  */
  is-path = builtins.isPath;

  /**
    Check if a value is a function.

    # Type

    ```
    is-function :: a -> Bool
    ```

    # Arguments

    - x: value to check

    # Examples

    ```nix
    is-function (x: x + 1)
    => true

    is-function map
    => true

    is-function 42
    => false
    ```
  */
  is-function = builtins.isFunction;

  /**
    Get the type of a value as a string.

    Returns one of: "int", "bool", "string", "path", "null", "set",
    "list", "lambda", or "float".

    # Type

    ```
    typeof :: a -> String
    ```

    # Arguments

    - x: value to check

    # Examples

    ```nix
    typeof 42
    => "int"

    typeof "hello"
    => "string"

    typeof { a = 1; }
    => "set"

    typeof [ 1 2 ]
    => "list"

    typeof (x: x)
    => "lambda"
    ```
  */
  typeof = builtins.typeOf;

}
