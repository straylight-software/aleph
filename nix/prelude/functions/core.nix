# nix/prelude/functions/core.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                 // core //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     And in the bloodlit dark behind his eyes, silver phosphenes
#     boiling in from the edge of space, hypnagogic images jerking
#     past like film compiled from random frames. Symbols, figures,
#     faces, a blurred, fragmented mandala of visual information.
#
#                                                         — Neuromancer
#
# Fundamentals and list operations. id, const, compose, map, fold —
# the primitives everything else is built from.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
rec {
  # ─────────────────────────────────────────────────────────────────────────
  # Fundamentals
  # ─────────────────────────────────────────────────────────────────────────

  /**
    The identity function. Returns its argument unchanged.

    Useful as a default function argument or in higher-order contexts
    where a "do nothing" transformation is needed.

    # Type

    ```
    id :: a -> a
    ```

    # Arguments

    - x: any value

    # Examples

    ```nix
    id 42
    => 42

    id "hello"
    => "hello"

    map id [ 1 2 3 ]
    => [ 1 2 3 ]
    ```
  */
  id = x: x;

  /**
    Create a constant function that always returns the first argument.

    Takes two arguments and returns the first, ignoring the second.
    Useful for creating functions that ignore their input.

    # Type

    ```
    const :: a -> b -> a
    ```

    # Arguments

    - a: the value to always return
    - b: ignored argument

    # Examples

    ```nix
    const 42 "ignored"
    => 42

    map (const 0) [ "a" "b" "c" ]
    => [ 0 0 0 ]

    const "default" null
    => "default"
    ```
  */
  const = a: _b: a;

  /**
    Flip the order of arguments to a binary function.

    Takes a function of two arguments and returns a new function
    with the argument order reversed.

    # Type

    ```
    flip :: (a -> b -> c) -> b -> a -> c
    ```

    # Arguments

    - f: binary function to flip
    - a: becomes the second argument to f
    - b: becomes the first argument to f

    # Examples

    ```nix
    flip builtins.sub 3 10
    => 7

    flip const "first" "second"
    => "second"

    (flip map [ 1 2 3 ]) (x: x * 2)
    => [ 2 4 6 ]
    ```
  */
  flip =
    f: a: b:
    f b a;

  /**
    Compose two functions (right to left).

    Creates a new function that applies g first, then f to the result.
    Mathematical notation: (f . g)(x) = f(g(x))

    # Type

    ```
    compose :: (b -> c) -> (a -> b) -> a -> c
    ```

    # Arguments

    - f: outer function to apply second
    - g: inner function to apply first
    - x: input value

    # Examples

    ```nix
    compose (x: x + 1) (x: x * 2) 5
    => 11

    compose toString (x: x * x) 4
    => "16"

    (compose head tail) [ 1 2 3 ]
    => 2
    ```
  */
  compose =
    f: g: x:
    f (g x);

  /**
    Compose two functions (left to right).

    Creates a new function that applies f first, then g to the result.
    Also known as forward composition or "then".

    # Type

    ```
    pipe :: (a -> b) -> (b -> c) -> a -> c
    ```

    # Arguments

    - f: first function to apply
    - g: second function to apply
    - x: input value

    # Examples

    ```nix
    pipe (x: x * 2) (x: x + 1) 5
    => 11

    pipe (x: x * x) toString 4
    => "16"

    (pipe tail head) [ 1 2 3 ]
    => 2
    ```
  */
  pipe =
    f: g: x:
    g (f x);

  /**
    Apply a function to a value.

    Simple function application. Useful in pipelines or when you need
    to apply a function stored in a variable.

    # Type

    ```
    apply :: (a -> b) -> a -> b
    ```

    # Arguments

    - f: function to apply
    - x: value to apply it to

    # Examples

    ```nix
    apply (x: x * 2) 21
    => 42

    apply toString 123
    => "123"
    ```
  */
  apply = f: f;

  /**
    Compute the fixed point of a function.

    Returns a value x such that f(x) = x. Used for defining recursive
    values and creating self-referential data structures.

    # Type

    ```
    fix :: (a -> a) -> a
    ```

    # Arguments

    - f: function whose fixed point to compute

    # Examples

    ```nix
    fix (self: { a = 1; b = self.a + 1; })
    => { a = 1; b = 2; }

    fix (fac: n: if n <= 1 then 1 else n * fac (n - 1)) 5
    => 120
    ```
  */
  fix =
    f:
    let
      x = f x;
    in
    x;

  /**
    Apply a binary operator to two values after transforming each.

    Transforms both arguments with f, then combines with op.
    Useful for comparing or combining values by a derived property.

    # Type

    ```
    on :: (b -> b -> c) -> (a -> b) -> a -> a -> c
    ```

    # Arguments

    - op: binary operator to apply to transformed values
    - f: transformation to apply to each argument
    - x: first value
    - y: second value

    # Examples

    ```nix
    on builtins.lessThan stringLength "hi" "hello"
    => true

    on add (x: x * x) 3 4
    => 25

    on (a: b: a == b) (x: x.id) { id = 1; } { id = 1; name = "foo"; }
    => true
    ```
  */
  on =
    op: f: x: y:
    op (f x) (f y);

  # ─────────────────────────────────────────────────────────────────────────
  # Lists (Part 1: Basic Operations)
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Apply a function to each element of a list.

    # Type

    ```
    map :: (a -> b) -> [a] -> [b]
    ```

    # Arguments

    - f: function to apply to each element
    - xs: input list

    # Examples

    ```nix
    map (x: x * 2) [ 1 2 3 ]
    => [ 2 4 6 ]
    ```
  */
  inherit (builtins) map;

  /**
    Select elements from a list that satisfy a predicate.

    # Type

    ```
    filter :: (a -> Bool) -> [a] -> [a]
    ```

    # Arguments

    - pred: predicate function returning true for elements to keep
    - xs: input list

    # Examples

    ```nix
    filter (x: x > 2) [ 1 2 3 4 5 ]
    => [ 3 4 5 ]
    ```
  */
  inherit (builtins) filter;

  /**
    Left fold over a list (strict). Reduces a list to a single value
    by applying a binary function from left to right.

    # Type

    ```
    fold :: (b -> a -> b) -> b -> [a] -> b
    ```

    # Arguments

    - f: binary function taking accumulator and element
    - init: initial accumulator value
    - xs: list to fold

    # Examples

    ```nix
    fold (acc: x: acc + x) 0 [ 1 2 3 4 ]
    => 10
    ```
  */
  fold = lib.foldl';

  /**
    Right fold over a list. Reduces a list to a single value
    by applying a binary function from right to left.

    # Type

    ```
    fold-right :: (a -> b -> b) -> b -> [a] -> b
    ```

    # Arguments

    - f: binary function taking element and accumulator
    - init: initial accumulator value
    - xs: list to fold

    # Examples

    ```nix
    fold-right (x: acc: [ x ] ++ acc) [ ] [ 1 2 3 ]
    => [ 1 2 3 ]
    ```
  */
  fold-right = lib.foldr;

  /**
    Return the first element of a non-empty list.

    # Type

    ```
    head :: [a] -> a
    ```

    # Arguments

    - xs: non-empty list

    # Examples

    ```nix
    head [ 1 2 3 ]
    => 1
    ```
  */
  inherit (builtins) head;

  /**
    Return all elements except the first of a non-empty list.

    # Type

    ```
    tail :: [a] -> [a]
    ```

    # Arguments

    - xs: non-empty list

    # Examples

    ```nix
    tail [ 1 2 3 ]
    => [ 2 3 ]
    ```
  */
  inherit (builtins) tail;

  /**
    Return all elements except the last of a non-empty list.

    # Type

    ```
    init :: [a] -> [a]
    ```

    # Arguments

    - xs: non-empty list

    # Examples

    ```nix
    init [ 1 2 3 ]
    => [ 1 2 ]
    ```
  */
  inherit (lib) init;

  /**
    Return the last element of a non-empty list.

    # Type

    ```
    last :: [a] -> a
    ```

    # Arguments

    - xs: non-empty list

    # Examples

    ```nix
    last [ 1 2 3 ]
    => 3
    ```
  */
  inherit (lib) last;

  /**
    Return the first n elements of a list.

    Negative values are clamped to 0 for safety.

    # Type

    ```
    take :: Int -> [a] -> [a]
    ```

    # Arguments

    - n: number of elements to take (clamped to >= 0)
    - xs: input list

    # Examples

    ```nix
    take 2 [ 1 2 3 4 5 ]
    => [ 1 2 ]

    take (-5) [ 1 2 3 ]
    => [ ]
    ```
  */
  take = n: xs: lib.take (if n < 0 then 0 else n) xs;

  /**
    Return the list without the first n elements.

    Negative values are clamped to 0 for safety.

    # Type

    ```
    drop :: Int -> [a] -> [a]
    ```

    # Arguments

    - n: number of elements to drop (clamped to >= 0)
    - xs: input list

    # Examples

    ```nix
    drop 2 [ 1 2 3 4 5 ]
    => [ 3 4 5 ]

    drop (-5) [ 1 2 3 ]
    => [ 1 2 3 ]
    ```
  */
  drop = n: xs: lib.drop (if n < 0 then 0 else n) xs;

  /**
    Return the number of elements in a list.

    # Type

    ```
    length :: [a] -> Int
    ```

    # Arguments

    - xs: input list

    # Examples

    ```nix
    length [ 1 2 3 ]
    => 3
    ```
  */
  inherit (builtins) length;

  /**
    Reverse the order of elements in a list.

    # Type

    ```
    reverse :: [a] -> [a]
    ```

    # Arguments

    - xs: input list

    # Examples

    ```nix
    reverse [ 1 2 3 ]
    => [ 3 2 1 ]
    ```
  */
  reverse = lib.reverseList;

  /**
    Concatenate a list of lists into a single list.

    # Type

    ```
    concat :: [[a]] -> [a]
    ```

    # Arguments

    - xss: list of lists

    # Examples

    ```nix
    concat [ [ 1 2 ] [ 3 4 ] [ 5 ] ]
    => [ 1 2 3 4 5 ]
    ```
  */
  concat = lib.concatLists;

  /**
    Flatten an arbitrarily nested list structure into a flat list.

    # Type

    ```
    flatten :: NestedList -> [a]
    ```

    # Arguments

    - xs: arbitrarily nested list structure

    # Examples

    ```nix
    flatten [ 1 [ 2 [ 3 4 ] 5 ] 6 ]
    => [ 1 2 3 4 5 6 ]
    ```
  */
  inherit (lib) flatten;

  /**
    Map a function over a list and concatenate the results.

    # Type

    ```
    concat-map :: (a -> [b]) -> [a] -> [b]
    ```

    # Arguments

    - f: function returning a list for each element
    - xs: input list

    # Examples

    ```nix
    concat-map (x: [ x (x * 10) ]) [ 1 2 3 ]
    => [ 1 10 2 20 3 30 ]
    ```
  */
  concat-map = lib.concatMap;

  # ─────────────────────────────────────────────────────────────────────────
  # Lists (Part 2: Zip, Sort, Search)
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Combine two lists into a list of pairs.

    Zips two lists together element-wise, producing a list of attribute sets
    with `fst` and `snd` fields. The result length is the minimum of the
    two input lengths.

    # Type

    ```
    zip :: [a] -> [b] -> [{fst: a, snd: b}]
    ```

    # Arguments

    - xs: first list to zip
    - ys: second list to zip

    # Examples

    ```nix
    zip [1 2 3] ["a" "b" "c"]
    => [{fst = 1; snd = "a";} {fst = 2; snd = "b";} {fst = 3; snd = "c";}]

    zip [1 2] ["a" "b" "c"]
    => [{fst = 1; snd = "a";} {fst = 2; snd = "b";}]
    ```
  */
  zip = lib.zipLists;

  /**
    Combine two lists using a custom function.

    Zips two lists together, applying a binary function to combine
    corresponding elements. The result length is the minimum of the
    two input lengths.

    # Type

    ```
    zip-with :: (a -> b -> c) -> [a] -> [b] -> [c]
    ```

    # Arguments

    - f: function to combine elements from both lists
    - xs: first list
    - ys: second list

    # Examples

    ```nix
    zip-with (a: b: a + b) [1 2 3] [10 20 30]
    => [11 22 33]

    zip-with (a: b: {x = a; y = b;}) ["a" "b"] [1 2]
    => [{x = "a"; y = 1;} {x = "b"; y = 2;}]
    ```
  */
  zip-with = lib.zipListsWith;

  /**
    Sort a list using a comparison function.

    Sorts list elements using a comparator function that returns true
    if the first argument should come before the second.

    # Type

    ```
    sort :: (a -> a -> Bool) -> [a] -> [a]
    ```

    # Arguments

    - cmp: comparison function returning true if first arg precedes second
    - xs: list to sort

    # Examples

    ```nix
    sort (a: b: a < b) [3 1 2]
    => [1 2 3]

    sort (a: b: a > b) [3 1 2]
    => [3 2 1]
    ```
  */
  inherit (builtins) sort;

  /**
    Sort a list by a derived key.

    Sorts elements by comparing the results of applying a key function
    to each element. Equivalent to `sort (a: b: f a < f b)`.

    # Type

    ```
    sort-on :: (a -> Ord) -> [a] -> [a]
    ```

    # Arguments

    - f: function to extract a comparable key from each element
    - xs: list to sort

    # Examples

    ```nix
    sort-on (x: x.age) [{name = "bob"; age = 30;} {name = "alice"; age = 25;}]
    => [{name = "alice"; age = 25;} {name = "bob"; age = 30;}]

    sort-on builtins.stringLength ["aaa" "b" "cc"]
    => ["b" "cc" "aaa"]
    ```
  */
  sort-on = f: builtins.sort (a: b: f a < f b);

  /**
    Remove duplicate elements from a list.

    Returns a list with duplicate elements removed, preserving the
    first occurrence of each element. Comparison is by value equality.

    # Type

    ```
    unique :: [a] -> [a]
    ```

    # Arguments

    - xs: list potentially containing duplicates

    # Examples

    ```nix
    unique [1 2 1 3 2 1]
    => [1 2 3]

    unique ["a" "b" "a"]
    => ["a" "b"]
    ```
  */
  inherit (lib) unique;

  /**
    Test if an element is in a list.

    Returns true if the element is present in the list (by value equality).

    # Type

    ```
    elem :: a -> [a] -> Bool
    ```

    # Arguments

    - x: element to search for
    - xs: list to search in

    # Examples

    ```nix
    elem 2 [1 2 3]
    => true

    elem 4 [1 2 3]
    => false
    ```
  */
  inherit (builtins) elem;

  /**
    Find the first element matching a predicate.

    Returns the first element satisfying the predicate, or the default
    value if no element matches.

    # Type

    ```
    find :: (a -> Bool) -> a -> [a] -> a
    ```

    # Arguments

    - pred: predicate function
    - default: value to return if no element matches
    - xs: list to search

    # Examples

    ```nix
    find (x: x > 2) 0 [1 2 3 4]
    => 3

    find (x: x > 10) 0 [1 2 3 4]
    => 0
    ```
  */
  find = lib.findFirst;

  /**
    Partition a list by a predicate.

    Splits a list into two lists: elements satisfying the predicate
    go into `right`, others go into `wrong`.

    # Type

    ```
    partition :: (a -> Bool) -> [a] -> {right: [a], wrong: [a]}
    ```

    # Arguments

    - pred: predicate to test each element
    - xs: list to partition

    # Examples

    ```nix
    partition (x: x > 2) [1 2 3 4]
    => {right = [3 4]; wrong = [1 2];}

    partition (x: x mod 2 == 0) [1 2 3 4 5 6]
    => {right = [2 4 6]; wrong = [1 3 5];}
    ```
  */
  inherit (lib) partition;

  /**
    Group list elements by a string key.

    Groups elements into an attribute set where keys are computed by
    applying the key function to each element. Elements with the same
    key are collected into lists.

    # Type

    ```
    group-by :: (a -> String) -> [a] -> {String: [a]}
    ```

    # Arguments

    - f: function to compute a string key for each element
    - xs: list to group

    # Examples

    ```nix
    group-by (x: x.type) [{type = "a"; v = 1;} {type = "b"; v = 2;} {type = "a"; v = 3;}]
    => {a = [{type = "a"; v = 1;} {type = "a"; v = 3;}]; b = [{type = "b"; v = 2;}];}

    group-by (x: toString (x mod 2)) [1 2 3 4 5]
    => {"0" = [2 4]; "1" = [1 3 5];}
    ```
  */
  group-by = builtins.groupBy;

  /**
    Generate a list of integers in a range.

    Returns a list of integers from `first` to `last` inclusive.
    If `first > last`, returns an empty list.

    # Type

    ```
    range :: Int -> Int -> [Int]
    ```

    # Arguments

    - first: starting integer (inclusive)
    - last: ending integer (inclusive)

    # Examples

    ```nix
    range 1 5
    => [1 2 3 4 5]

    range 0 3
    => [0 1 2 3]

    range 5 3
    => []
    ```
  */
  inherit (lib) range;

  /**
    Create a list with n copies of a value.

    Returns a list containing the given value repeated n times.

    # Type

    ```
    replicate :: Int -> a -> [a]
    ```

    # Arguments

    - n: number of copies
    - x: value to replicate

    # Examples

    ```nix
    replicate 3 "x"
    => ["x" "x" "x"]

    replicate 4 0
    => [0 0 0 0]

    replicate 0 "x"
    => []
    ```
  */
  inherit (lib) replicate;

  /**
    Insert a separator element between each element of a list.

    # Type

    ```
    intersperse :: a -> [a] -> [a]
    ```

    # Arguments

    - sep: separator element to insert between elements
    - xs: input list

    # Examples

    ```nix
    intersperse 0 [ 1 2 3 ]
    => [ 1 0 2 0 3 ]
    ```
  */
  intersperse =
    sep: xs:
    if xs == [ ] then
      [ ]
    else
      [ (builtins.head xs) ]
      ++ lib.concatMap (x: [
        sep
        x
      ]) (builtins.tail xs);

  /**
    Left-to-right scan, returning all intermediate accumulator states.

    Similar to foldl, but returns a list of successive reduced values from the left.
    The result list length is always one more than the input list length.

    # Type

    ```
    scanl :: (b -> a -> b) -> b -> [a] -> [b]
    ```

    # Arguments

    - f: binary function taking accumulator and element, returning new accumulator
    - init: initial accumulator value
    - xs: input list

    # Examples

    ```nix
    scanl (a: b: a + b) 0 [ 1 2 3 ]
    => [ 0 1 3 6 ]
    ```
  */
  scanl =
    f: init: xs:
    lib.foldl' (acc: x: acc ++ [ (f (lib.last acc) x) ]) [ init ] xs;

  /**
    Split a list into the longest prefix satisfying a predicate and the remainder.

    Returns an attribute set with `fst` containing the longest prefix of elements
    satisfying the predicate, and `snd` containing the rest of the list.

    # Type

    ```
    span :: (a -> Bool) -> [a] -> { fst :: [a]; snd :: [a]; }
    ```

    # Arguments

    - pred: predicate function to test elements
    - xs: input list

    # Examples

    ```nix
    span (x: x < 3) [ 1 2 3 4 1 2 ]
    => { fst = [ 1 2 ]; snd = [ 3 4 1 2 ]; }
    ```
  */
  span = pred: xs: {
    fst = lib.takeWhile pred xs;
    snd = lib.dropWhile pred xs;
  };

  /**
    Split a list at the first element satisfying a predicate.

    Equivalent to `span` with a negated predicate. Returns an attribute set with
    `fst` containing elements before the first match, and `snd` containing the
    rest starting from the first match.

    # Type

    ```
    break :: (a -> Bool) -> [a] -> { fst :: [a]; snd :: [a]; }
    ```

    # Arguments

    - pred: predicate function to test elements
    - xs: input list

    # Examples

    ```nix
    break (x: x >= 3) [ 1 2 3 4 1 2 ]
    => { fst = [ 1 2 ]; snd = [ 3 4 1 2 ]; }
    ```
  */
  break = pred: xs: span (x: !(pred x)) xs;

  /**
    Split a list at a given index position.

    Returns an attribute set with `fst` containing the first n elements,
    and `snd` containing the remaining elements.

    # Type

    ```
    split-at :: Int -> [a] -> { fst :: [a]; snd :: [a]; }
    ```

    # Arguments

    - n: index at which to split (0-based)
    - xs: input list

    # Examples

    ```nix
    split-at 2 [ 1 2 3 4 5 ]
    => { fst = [ 1 2 ]; snd = [ 3 4 5 ]; }
    ```
  */
  split-at = n: xs: {
    fst = lib.take n xs;
    snd = lib.drop n xs;
  };

  /**
    Find the minimum element in a non-empty list using default comparison.

    Throws with a clear message for empty lists.

    # Type

    ```
    minimum :: [Ord] -> Ord
    ```

    # Arguments

    - xs: non-empty list of comparable elements

    # Examples

    ```nix
    minimum [ 3 1 4 1 5 9 ]
    => 1

    minimum [ ]
    => error: minimum: empty list has no minimum
    ```
  */
  minimum =
    xs:
    if xs == [ ] then
      throw "minimum: empty list has no minimum"
    else
      lib.foldl' (a: b: if a < b then a else b) (builtins.head xs) (builtins.tail xs);

  /**
    Find the maximum element in a non-empty list using default comparison.

    Throws with a clear message for empty lists.

    # Type

    ```
    maximum :: [Ord] -> Ord
    ```

    # Arguments

    - xs: non-empty list of comparable elements

    # Examples

    ```nix
    maximum [ 3 1 4 1 5 9 ]
    => 9

    maximum [ ]
    => error: maximum: empty list has no maximum
    ```
  */
  maximum =
    xs:
    if xs == [ ] then
      throw "maximum: empty list has no maximum"
    else
      lib.foldl' (a: b: if a > b then a else b) (builtins.head xs) (builtins.tail xs);

  /**
    Find the minimum element in a non-empty list by comparing projected values.

    Returns the original element (not the projected value) that has the smallest
    projection. The list must be non-empty; behavior is undefined for empty lists.

    # Type

    ```
    minimum-by :: (a -> Ord) -> [a] -> a
    ```

    # Arguments

    - f: projection function to extract comparison key
    - xs: non-empty list of elements

    # Examples

    ```nix
    minimum-by (x: x.age) [ { name = "alice"; age = 30; } { name = "bob"; age = 25; } ]
    => { name = "bob"; age = 25; }
    ```
  */
  minimum-by =
    f: xs: lib.foldl' (a: b: if (f a) < (f b) then a else b) (builtins.head xs) (builtins.tail xs);

  /**
    Find the maximum element in a non-empty list by comparing projected values.

    Returns the original element (not the projected value) that has the largest
    projection. The list must be non-empty; behavior is undefined for empty lists.

    # Type

    ```
    maximum-by :: (a -> Ord) -> [a] -> a
    ```

    # Arguments

    - f: projection function to extract comparison key
    - xs: non-empty list of elements

    # Examples

    ```nix
    maximum-by (x: x.age) [ { name = "alice"; age = 30; } { name = "bob"; age = 25; } ]
    => { name = "alice"; age = 30; }
    ```
  */
  maximum-by =
    f: xs: lib.foldl' (a: b: if (f a) > (f b) then a else b) (builtins.head xs) (builtins.tail xs);

  /**
    Alias for `minimum-by`. Find the minimum element by comparing projected values.

    # Type

    ```
    minimum-on :: (a -> Ord) -> [a] -> a
    ```

    # Arguments

    - f: projection function to extract comparison key
    - xs: non-empty list of elements

    # Examples

    ```nix
    minimum-on (x: x.len) [ { name = "ab"; len = 2; } { name = "a"; len = 1; } ]
    => { name = "a"; len = 1; }
    ```
  */
  minimum-on = minimum-by;

  /**
    Alias for `maximum-by`. Find the maximum element by comparing projected values.

    # Type

    ```
    maximum-on :: (a -> Ord) -> [a] -> a
    ```

    # Arguments

    - f: projection function to extract comparison key
    - xs: non-empty list of elements

    # Examples

    ```nix
    maximum-on (x: x.len) [ { name = "ab"; len = 2; } { name = "a"; len = 1; } ]
    => { name = "ab"; len = 2; }
    ```
  */
  maximum-on = maximum-by;

  # ─────────────────────────────────────────────────────────────────────────
  # Lists (Part 4: Index and Modification Operations)
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Find the index of the first element satisfying a predicate.

    Returns the zero-based index of the first element for which the predicate
    returns true, or null if no such element exists.

    # Type

    ```
    find-index :: (a -> Bool) -> [a] -> Int | Null
    ```

    # Arguments

    - pred: predicate function to test each element
    - xs: list to search

    # Examples

    ```nix
    find-index (x: x > 5) [ 1 3 7 2 9 ]
    => 2

    find-index (x: x > 100) [ 1 2 3 ]
    => null
    ```
  */
  find-index =
    pred: xs:
    let
      go =
        i: xs:
        if xs == [ ] then
          null
        else if pred (builtins.head xs) then
          i
        else
          go (i + 1) (builtins.tail xs);
    in
    go 0 xs;

  /**
    Find all indices of elements satisfying a predicate.

    Returns a list of zero-based indices for all elements where the predicate
    returns true. Returns an empty list if no elements match.

    # Type

    ```
    find-indices :: (a -> Bool) -> [a] -> [Int]
    ```

    # Arguments

    - pred: predicate function to test each element
    - xs: list to search

    # Examples

    ```nix
    find-indices (x: x > 3) [ 1 5 2 7 3 9 ]
    => [ 1 3 5 ]

    find-indices (x: x < 0) [ 1 2 3 ]
    => [ ]
    ```
  */
  find-indices =
    pred: xs:
    lib.foldl' (acc: i: if pred (builtins.elemAt xs i) then acc ++ [ i ] else acc) [ ] (
      lib.range 0 (builtins.length xs - 1)
    );

  /**
    Find the index of the first occurrence of an element.

    Returns the zero-based index of the first element equal to the target,
    or null if the element is not found.

    # Type

    ```
    elem-index :: a -> [a] -> Int | Null
    ```

    # Arguments

    - x: element to search for
    - xs: list to search

    # Examples

    ```nix
    elem-index "b" [ "a" "b" "c" "b" ]
    => 1

    elem-index "z" [ "a" "b" "c" ]
    => null
    ```
  */
  elem-index =
    x: xs:
    let
      go =
        i: xs:
        if xs == [ ] then
          null
        else if builtins.head xs == x then
          i
        else
          go (i + 1) (builtins.tail xs);
    in
    go 0 xs;

  /**
    Find all indices of occurrences of an element.

    Returns a list of zero-based indices for all positions where the element
    occurs. Returns an empty list if the element is not found.

    # Type

    ```
    elem-indices :: a -> [a] -> [Int]
    ```

    # Arguments

    - x: element to search for
    - xs: list to search

    # Examples

    ```nix
    elem-indices "b" [ "a" "b" "c" "b" "d" "b" ]
    => [ 1 3 5 ]

    elem-indices "z" [ "a" "b" "c" ]
    => [ ]
    ```
  */
  elem-indices =
    x: xs:
    lib.foldl' (acc: i: if builtins.elemAt xs i == x then acc ++ [ i ] else acc) [ ] (
      lib.range 0 (builtins.length xs - 1)
    );

  /**
    Update the element at a given index by applying a function.

    Applies the function to the element at index i and returns a new list
    with the result at that position. Returns the original list unchanged
    if the index is out of bounds.

    # Type

    ```
    update-at :: Int -> (a -> a) -> [a] -> [a]
    ```

    # Arguments

    - i: zero-based index of element to update
    - f: function to apply to the element
    - xs: input list

    # Examples

    ```nix
    update-at 1 (x: x * 10) [ 1 2 3 4 ]
    => [ 1 20 3 4 ]

    update-at 0 (s: "Hello, " + s) [ "world" "!" ]
    => [ "Hello, world" "!" ]

    update-at 99 (x: x + 1) [ 1 2 3 ]
    => [ 1 2 3 ]
    ```
  */
  update-at =
    i: f: xs:
    let
      len = builtins.length xs;
    in
    if i < 0 || i >= len then
      xs
    else
      lib.take i xs ++ [ (f (builtins.elemAt xs i)) ] ++ lib.drop (i + 1) xs;

  /**
    Set the element at a given index to a new value.

    Returns a new list with the element at index i replaced by the new value.
    Returns the original list unchanged if the index is out of bounds.

    # Type

    ```
    set-at :: Int -> a -> [a] -> [a]
    ```

    # Arguments

    - i: zero-based index of element to replace
    - x: new value
    - xs: input list

    # Examples

    ```nix
    set-at 2 99 [ 1 2 3 4 5 ]
    => [ 1 2 99 4 5 ]

    set-at 0 "first" [ "a" "b" "c" ]
    => [ "first" "b" "c" ]

    set-at 10 "x" [ "a" "b" ]
    => [ "a" "b" ]
    ```
  */
  set-at =
    i: x: xs:
    let
      len = builtins.length xs;
    in
    if i < 0 || i >= len then xs else lib.take i xs ++ [ x ] ++ lib.drop (i + 1) xs;

  /**
    Insert an element at a given index.

    Returns a new list with the element inserted at index i. Elements at
    and after index i are shifted right. If i is greater than the list
    length, the element is appended at the end.

    # Type

    ```
    insert-at :: Int -> a -> [a] -> [a]
    ```

    # Arguments

    - i: zero-based index where element should be inserted
    - x: element to insert
    - xs: input list

    # Examples

    ```nix
    insert-at 2 99 [ 1 2 3 4 ]
    => [ 1 2 99 3 4 ]

    insert-at 0 "first" [ "a" "b" ]
    => [ "first" "a" "b" ]

    insert-at 10 "end" [ "a" "b" ]
    => [ "a" "b" "end" ]
    ```
  */
  insert-at =
    i: x: xs:
    lib.take i xs ++ [ x ] ++ lib.drop i xs;

  /**
    Remove the element at a given index.

    Returns a new list with the element at index i removed. Elements after
    index i are shifted left. If the index is out of bounds, behavior
    depends on take/drop semantics (safe for positive indices).

    # Type

    ```
    remove-at :: Int -> [a] -> [a]
    ```

    # Arguments

    - i: zero-based index of element to remove
    - xs: input list

    # Examples

    ```nix
    remove-at 2 [ 1 2 3 4 5 ]
    => [ 1 2 4 5 ]

    remove-at 0 [ "a" "b" "c" ]
    => [ "b" "c" ]

    remove-at 1 [ "only" "two" ]
    => [ "only" ]
    ```
  */
  remove-at = i: xs: lib.take i xs ++ lib.drop (i + 1) xs;

  /**
    Split a list into chunks of a given size.

    Returns a list of sublists, each containing at most n elements.
    The last chunk may contain fewer elements if the list length is
    not evenly divisible by n.

    # Type

    ```
    chunks-of :: Int -> [a] -> [[a]]
    ```

    # Arguments

    - n: maximum size of each chunk (must be positive)
    - xs: list to split

    # Examples

    ```nix
    chunks-of 2 [ 1 2 3 4 5 ]
    => [ [ 1 2 ] [ 3 4 ] [ 5 ] ]

    chunks-of 3 [ "a" "b" "c" "d" "e" "f" ]
    => [ [ "a" "b" "c" ] [ "d" "e" "f" ] ]

    chunks-of 10 [ 1 2 3 ]
    => [ [ 1 2 3 ] ]

    chunks-of 2 [ ]
    => [ ]
    ```
  */
  chunks-of =
    n: xs:
    # Guard against n <= 0 to prevent infinite recursion
    if n <= 0 then
      throw "chunks-of: chunk size must be positive, got ${toString n}"
    else if xs == [ ] then
      [ ]
    else
      [ (lib.take n xs) ] ++ chunks-of n (lib.drop n xs);

  /**
    Remove duplicate elements based on a key function.

    Returns a list with duplicates removed, where duplicates are determined
    by comparing the result of applying the key function to each element.
    Keeps the first occurrence of each unique key.

    # Type

    ```
    unique-by :: (a -> b) -> [a] -> [a]
    ```

    # Arguments

    - f: key extraction function
    - xs: input list

    # Examples

    ```nix
    unique-by (x: x.id) [ { id = 1; v = "a"; } { id = 2; v = "b"; } { id = 1; v = "c"; } ]
    => [ { id = 1; v = "a"; } { id = 2; v = "b"; } ]

    unique-by lib.toLower [ "Hello" "WORLD" "hello" "World" ]
    => [ "Hello" "WORLD" ]

    unique-by (x: x mod 3) [ 1 2 3 4 5 6 ]
    => [ 1 2 3 ]
    ```
  */
  unique-by =
    f: xs:
    let
      go =
        seen: xs:
        if xs == [ ] then
          [ ]
        else
          let
            x = builtins.head xs;
            rest = builtins.tail xs;
            key = f x;
          in
          if builtins.elem key seen then go seen rest else [ x ] ++ go (seen ++ [ key ]) rest;
    in
    go [ ] xs;
}
