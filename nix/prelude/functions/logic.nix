# nix/prelude/functions/logic.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // logic //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'The Turing heat,' he said. 'The ice. It's the reason I'm
#      here. I'm good at what I do.' The anger was gone now,
#      replaced by a kind of weary patience. 'The best.'
#
#                                                         — Neuromancer
#
# Comparison, boolean, and arithmetic. The logic primitives.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
rec {
  # ─────────────────────────────────────────────────────────────────────────
  # Comparison
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Test two values for equality.

    # Type

    ```
    eq :: a -> a -> Bool
    ```

    # Arguments

    - a: first value
    - b: second value

    # Examples

    ```nix
    eq 1 1
    => true

    eq "foo" "bar"
    => false
    ```
  */
  eq = a: b: a == b;

  /**
    Test two values for inequality.

    # Type

    ```
    neq :: a -> a -> Bool
    ```

    # Arguments

    - a: first value
    - b: second value

    # Examples

    ```nix
    neq 1 2
    => true

    neq "foo" "foo"
    => false
    ```
  */
  neq = a: b: a != b;

  # Alias for neq (README uses `ne`)
  ne = neq;

  /**
    Test if the first value is strictly less than the second.

    # Type

    ```
    lt :: Ord a => a -> a -> Bool
    ```

    # Arguments

    - a: first value
    - b: second value

    # Examples

    ```nix
    lt 1 2
    => true

    lt 2 2
    => false
    ```
  */
  lt = a: b: a < b;

  /**
    Test if the first value is less than or equal to the second.

    # Type

    ```
    le :: Ord a => a -> a -> Bool
    ```

    # Arguments

    - a: first value
    - b: second value

    # Examples

    ```nix
    le 1 2
    => true

    le 2 2
    => true

    le 3 2
    => false
    ```
  */
  le = a: b: a <= b;

  /**
    Test if the first value is strictly greater than the second.

    # Type

    ```
    gt :: Ord a => a -> a -> Bool
    ```

    # Arguments

    - a: first value
    - b: second value

    # Examples

    ```nix
    gt 2 1
    => true

    gt 2 2
    => false
    ```
  */
  gt = a: b: a > b;

  /**
    Test if the first value is greater than or equal to the second.

    # Type

    ```
    ge :: Ord a => a -> a -> Bool
    ```

    # Arguments

    - a: first value
    - b: second value

    # Examples

    ```nix
    ge 2 1
    => true

    ge 2 2
    => true

    ge 1 2
    => false
    ```
  */
  ge = a: b: a >= b;

  /**
    Return the smaller of two values.

    # Type

    ```
    min :: Ord a => a -> a -> a
    ```

    # Arguments

    - a: first value
    - b: second value

    # Examples

    ```nix
    min 1 2
    => 1

    min "apple" "banana"
    => "apple"
    ```
  */
  min = a: b: if a < b then a else b;

  /**
    Return the larger of two values.

    # Type

    ```
    max :: Ord a => a -> a -> a
    ```

    # Arguments

    - a: first value
    - b: second value

    # Examples

    ```nix
    max 1 2
    => 2

    max "apple" "banana"
    => "banana"
    ```
  */
  max = a: b: if a > b then a else b;

  /**
    Compare two values, returning -1, 0, or 1.

    Returns -1 if a < b, 1 if a > b, and 0 if a == b.

    # Type

    ```
    compare :: Ord a => a -> a -> Int
    ```

    # Arguments

    - a: first value
    - b: second value

    # Examples

    ```nix
    compare 1 2
    => -1

    compare 2 2
    => 0

    compare 3 2
    => 1
    ```
  */
  compare =
    a: b:
    if a < b then
      (-1)
    else if a > b then
      1
    else
      0;

  /**
    Clamp a value to a range.

    Returns lo if x < lo, hi if x > hi, otherwise x.

    # Type

    ```
    clamp :: Ord a => a -> a -> a -> a
    ```

    # Arguments

    - lo: lower bound
    - hi: upper bound
    - x: value to clamp

    # Examples

    ```nix
    clamp 0 10 5
    => 5

    clamp 0 10 (-5)
    => 0

    clamp 0 10 15
    => 10
    ```
  */
  clamp =
    lo: hi: x:
    if x < lo then
      lo
    else if x > hi then
      hi
    else
      x;

  /**
    Compare two values by applying a function first.

    Useful for sorting or comparing complex values by a derived key.

    # Type

    ```
    comparing :: (a -> b) -> a -> a -> Int
    ```

    # Arguments

    - f: function to apply before comparing
    - a: first value
    - b: second value

    # Examples

    ```nix
    comparing (x: x.age) { age = 25; } { age = 30; }
    => -1

    comparing stringLength "hi" "hello"
    => -1
    ```
  */
  comparing =
    f: a: b:
    compare (f a) (f b);

  # ─────────────────────────────────────────────────────────────────────────
  # Boolean
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Logical negation.

    # Type

    ```
    not :: Bool -> Bool
    ```

    # Arguments

    - x: boolean value to negate

    # Examples

    ```nix
    not true
    => false

    not false
    => true
    ```
  */
  not = x: !x;

  /**
    Logical conjunction (AND).

    # Type

    ```
    and :: Bool -> Bool -> Bool
    ```

    # Arguments

    - a: first boolean
    - b: second boolean

    # Examples

    ```nix
    and true true
    => true

    and true false
    => false
    ```
  */
  and = a: b: a && b;

  /**
    Logical disjunction (OR).

    # Type

    ```
    or :: Bool -> Bool -> Bool
    ```

    # Arguments

    - a: first boolean
    - b: second boolean

    # Examples

    ```nix
    or true false
    => true

    or false false
    => false
    ```
  */
  or = a: b: a || b;

  /**
    Test if all elements of a list satisfy a predicate.

    # Type

    ```
    all :: (a -> Bool) -> [a] -> Bool
    ```

    # Arguments

    - pred: predicate function
    - xs: list to test

    # Examples

    ```nix
    all (x: x > 0) [ 1 2 3 ]
    => true

    all (x: x > 0) [ 1 (-2) 3 ]
    => false
    ```
  */
  inherit (lib) all;

  /**
    Test if any element of a list satisfies a predicate.

    # Type

    ```
    any :: (a -> Bool) -> [a] -> Bool
    ```

    # Arguments

    - pred: predicate function
    - xs: list to test

    # Examples

    ```nix
    any (x: x > 2) [ 1 2 3 ]
    => true

    any (x: x > 5) [ 1 2 3 ]
    => false
    ```
  */
  inherit (lib) any;

  /**
    Test if no element of a list satisfies a predicate.

    The negation of `any`.

    # Type

    ```
    none :: (a -> Bool) -> [a] -> Bool
    ```

    # Arguments

    - pred: predicate function
    - xs: list to test

    # Examples

    ```nix
    none (x: x > 5) [ 1 2 3 ]
    => true

    none (x: x > 2) [ 1 2 3 ]
    => false
    ```
  */
  none = pred: xs: !(lib.any pred xs);

  /**
    Boolean fold: select between two values based on a condition.

    Similar to an if-then-else but with arguments in a different order,
    useful for partial application. Note: the false case comes first.

    # Type

    ```
    bool :: a -> a -> Bool -> a
    ```

    # Arguments

    - f: value to return if condition is false
    - t: value to return if condition is true
    - cond: boolean condition

    # Examples

    ```nix
    bool "no" "yes" true
    => "yes"

    bool "no" "yes" false
    => "no"

    map (bool 0 1) [ true false true ]
    => [ 1 0 1 ]
    ```
  */
  bool =
    f: t: cond:
    if cond then t else f;

  # ─────────────────────────────────────────────────────────────────────────
  # Arithmetic
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Add two numbers.

    # Type

    ```
    add :: Num -> Num -> Num
    ```

    # Arguments

    - a: first number
    - b: second number

    # Examples

    ```nix
    add 2 3
    => 5

    add 1.5 2.5
    => 4.0
    ```
  */
  add = a: b: a + b;

  /**
    Subtract second number from first.

    # Type

    ```
    sub :: Num -> Num -> Num
    ```

    # Arguments

    - a: number to subtract from
    - b: number to subtract

    # Examples

    ```nix
    sub 5 3
    => 2

    sub 1 4
    => -3
    ```
  */
  sub = a: b: a - b;

  /**
    Multiply two numbers.

    # Type

    ```
    mul :: Num -> Num -> Num
    ```

    # Arguments

    - a: first number
    - b: second number

    # Examples

    ```nix
    mul 3 4
    => 12

    mul 2.5 4
    => 10.0
    ```
  */
  mul = a: b: a * b;

  /**
    Divide first number by second.

    Integer division truncates toward zero for integers.

    # Type

    ```
    div :: Num -> Num -> Num
    ```

    # Arguments

    - a: dividend
    - b: divisor

    # Examples

    ```nix
    div 10 3
    => 3

    div 10.0 3
    => 3.333...
    ```
  */
  div = a: b: a / b;

  /**
    Compute remainder after integer division (modulo).

    # Type

    ```
    mod :: Int -> Int -> Int
    ```

    # Arguments

    - a: dividend
    - b: divisor

    # Examples

    ```nix
    mod 10 3
    => 1

    mod 15 5
    => 0
    ```
  */
  inherit (lib) mod;

  /**
    Negate a number.

    # Type

    ```
    neg :: Num -> Num
    ```

    # Arguments

    - x: number to negate

    # Examples

    ```nix
    neg 5
    => -5

    neg (-3)
    => 3
    ```
  */
  neg = x: -x;

  /**
    Compute absolute value of a number.

    # Type

    ```
    abs :: Num -> Num
    ```

    # Arguments

    - x: number

    # Examples

    ```nix
    abs (-5)
    => 5

    abs 3
    => 3

    abs 0
    => 0
    ```
  */
  abs = x: if x < 0 then -x else x;

  /**
    Return the sign of a number as -1, 0, or 1.

    # Type

    ```
    signum :: Num -> Int
    ```

    # Arguments

    - x: number to check

    # Examples

    ```nix
    signum (-5)
    => -1

    signum 0
    => 0

    signum 42
    => 1
    ```
  */
  signum =
    x:
    if x < 0 then
      (-1)
    else if x > 0 then
      1
    else
      0;

  /**
    Compute the sum of a list of numbers.

    # Type

    ```
    sum :: [Num] -> Num
    ```

    # Arguments

    - xs: list of numbers

    # Examples

    ```nix
    sum [ 1 2 3 4 5 ]
    => 15

    sum [ ]
    => 0
    ```
  */
  sum = lib.foldl' (a: b: a + b) 0;

  /**
    Compute the product of a list of numbers.

    # Type

    ```
    product :: [Num] -> Num
    ```

    # Arguments

    - xs: list of numbers

    # Examples

    ```nix
    product [ 1 2 3 4 ]
    => 24

    product [ ]
    => 1
    ```
  */
  product = lib.foldl' (a: b: a * b) 1;

}
