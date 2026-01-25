# The Prelude

A functional library for Nix. Provides 100+ functions with consistent naming and Haskell-style semantics.

## Access

Via flake module:

```nix
perSystem = { config, ... }:
  let inherit (config.aleph) prelude; in
  { ... }
```

Via overlay:

```nix
pkgs.aleph.prelude
```

## Functions by Category

### Fundamentals

| Function | Type | Description |
|----------|------|-------------|
| `id` | `a -> a` | Identity function |
| `const` | `a -> b -> a` | Constant function |
| `flip` | `(a -> b -> c) -> b -> a -> c` | Flip argument order |
| `compose` | `(b -> c) -> (a -> b) -> a -> c` | Right-to-left composition |
| `pipe` | `(a -> b) -> (b -> c) -> a -> c` | Left-to-right composition |
| `fix` | `(a -> a) -> a` | Fixed point combinator |
| `on` | `(b -> b -> c) -> (a -> b) -> a -> a -> c` | Apply function to both args |

### Lists

| Function | Type | Description |
|----------|------|-------------|
| `map` | `(a -> b) -> [a] -> [b]` | Transform each element |
| `filter` | `(a -> bool) -> [a] -> [a]` | Keep matching elements |
| `fold` | `(b -> a -> b) -> b -> [a] -> b` | Left fold (strict) |
| `fold-right` | `(a -> b -> b) -> b -> [a] -> b` | Right fold |
| `head` | `[a] -> a` | First element |
| `tail` | `[a] -> [a]` | All but first |
| `init` | `[a] -> [a]` | All but last |
| `last` | `[a] -> a` | Last element |
| `take` | `int -> [a] -> [a]` | First n elements |
| `drop` | `int -> [a] -> [a]` | Drop first n elements |
| `length` | `[a] -> int` | List length |
| `reverse` | `[a] -> [a]` | Reverse list |
| `concat` | `[[a]] -> [a]` | Flatten one level |
| `flatten` | `nested -> [a]` | Flatten all levels |
| `concat-map` | `(a -> [b]) -> [a] -> [b]` | Map and flatten |
| `zip` | `[a] -> [b] -> [{fst,snd}]` | Pair elements |
| `zip-with` | `(a -> b -> c) -> [a] -> [b] -> [c]` | Combine with function |
| `sort` | `(a -> a -> bool) -> [a] -> [a]` | Sort with comparator |
| `unique` | `[a] -> [a]` | Remove duplicates |
| `elem` | `a -> [a] -> bool` | Membership test |
| `find` | `(a -> bool) -> [a] -> a?` | First matching element |
| `partition` | `(a -> bool) -> [a] -> {right,wrong}` | Split by predicate |
| `group-by` | `(a -> string) -> [a] -> attrs` | Group by key function |
| `range` | `int -> int -> [int]` | Integer range |
| `replicate` | `int -> a -> [a]` | N copies of value |

### Attribute Sets

| Function | Type | Description |
|----------|------|-------------|
| `map-attrs` | `(string -> a -> b) -> attrs -> attrs` | Transform values |
| `filter-attrs` | `(string -> a -> bool) -> attrs -> attrs` | Keep matching |
| `fold-attrs` | `(b -> string -> a -> b) -> b -> attrs -> b` | Fold over attrs |
| `keys` | `attrs -> [string]` | Get all keys |
| `values` | `attrs -> [a]` | Get all values |
| `has` | `string -> attrs -> bool` | Key exists? |
| `get` | `[string] -> attrs -> a -> a` | Get nested with default |
| `get'` | `string -> attrs -> a` | Get (throws if missing) |
| `set` | `[string] -> a -> attrs -> attrs` | Set nested path |
| `remove` | `[string] -> attrs -> attrs` | Remove keys |
| `merge` | `attrs -> attrs -> attrs` | Shallow merge (right wins) |
| `merge-all` | `[attrs] -> attrs` | Merge list of attrs |
| `to-list` | `attrs -> [{name,value}]` | Convert to name-value pairs |
| `from-list` | `[{name,value}] -> attrs` | Convert from name-value pairs |
| `map-to-list` | `(string -> a -> b) -> attrs -> [b]` | Map and collect |
| `intersect` | `attrs -> attrs -> attrs` | Keys in both |
| `gen-attrs` | `[string] -> (string -> a) -> attrs` | Generate from names |

### Strings

| Function | Type | Description |
|----------|------|-------------|
| `split` | `string -> string -> [string]` | Split on separator |
| `join` | `string -> [string] -> string` | Join with separator |
| `trim` | `string -> string` | Remove whitespace |
| `replace` | `[string] -> [string] -> string -> string` | Replace substrings |
| `starts-with` | `string -> string -> bool` | Prefix test |
| `ends-with` | `string -> string -> bool` | Suffix test |
| `contains` | `string -> string -> bool` | Infix test |
| `to-lower` | `string -> string` | Lowercase |
| `to-upper` | `string -> string` | Uppercase |
| `to-string` | `a -> string` | Convert to string |
| `string-length` | `string -> int` | String length |
| `substring` | `int -> int -> string -> string` | Extract substring |

### Maybe (Null Handling)

| Function | Type | Description |
|----------|------|-------------|
| `maybe` | `b -> (a -> b) -> a? -> b` | Handle nullable with function |
| `from-maybe` | `a -> a? -> a` | Extract with default |
| `is-null` | `a? -> bool` | Check if null |
| `cat-maybes` | `[a?] -> [a]` | Filter out nulls |
| `map-maybe` | `(a -> b?) -> [a] -> [b]` | Map and filter nulls |

### Either (Error Handling)

| Function | Type | Description |
|----------|------|-------------|
| `left` | `a -> Either a b` | Create Left (error) |
| `right` | `b -> Either a b` | Create Right (success) |
| `is-left` | `Either a b -> bool` | Check if Left |
| `is-right` | `Either a b -> bool` | Check if Right |
| `either` | `(a -> c) -> (b -> c) -> Either a b -> c` | Handle both cases |
| `from-right` | `b -> Either a b -> b` | Extract Right with default |
| `from-left` | `a -> Either a b -> a` | Extract Left with default |

### Comparison & Boolean

| Function | Type | Description |
|----------|------|-------------|
| `eq` | `a -> a -> bool` | Equality |
| `neq` | `a -> a -> bool` | Inequality |
| `lt` | `a -> a -> bool` | Less than |
| `le` | `a -> a -> bool` | Less or equal |
| `gt` | `a -> a -> bool` | Greater than |
| `ge` | `a -> a -> bool` | Greater or equal |
| `min` | `a -> a -> a` | Minimum |
| `max` | `a -> a -> a` | Maximum |
| `compare` | `a -> a -> int` | Three-way compare |
| `clamp` | `a -> a -> a -> a` | Clamp to range |
| `not` | `bool -> bool` | Negation |
| `and` | `bool -> bool -> bool` | Conjunction |
| `or` | `bool -> bool -> bool` | Disjunction |
| `all` | `(a -> bool) -> [a] -> bool` | All satisfy predicate |
| `any` | `(a -> bool) -> [a] -> bool` | Any satisfies predicate |
| `bool` | `a -> a -> bool -> a` | Conditional value |

### Arithmetic

| Function | Type | Description |
|----------|------|-------------|
| `add` | `num -> num -> num` | Addition |
| `sub` | `num -> num -> num` | Subtraction |
| `mul` | `num -> num -> num` | Multiplication |
| `div` | `num -> num -> num` | Division |
| `mod` | `int -> int -> int` | Modulo |
| `neg` | `num -> num` | Negation |
| `abs` | `num -> num` | Absolute value |
| `sum` | `[num] -> num` | Sum of list |
| `product` | `[num] -> num` | Product of list |

### Type Predicates

| Function | Type | Description |
|----------|------|-------------|
| `is-list` | `a -> bool` | Is a list? |
| `is-attrs` | `a -> bool` | Is an attrset? |
| `is-string` | `a -> bool` | Is a string? |
| `is-int` | `a -> bool` | Is an integer? |
| `is-bool` | `a -> bool` | Is a boolean? |
| `is-float` | `a -> bool` | Is a float? |
| `is-path` | `a -> bool` | Is a path? |
| `is-function` | `a -> bool` | Is a function? |
| `typeof` | `a -> string` | Type as string |

## Example

```nix
let
  inherit (config.aleph.prelude) map filter fold from-maybe starts-with;

  # Filter and transform
  nixFiles = filter (f: ends-with ".nix" f) files;
  names = map (f: removeSuffix ".nix" f) nixFiles;

  # Fold with accumulator
  total = fold (acc: x: acc + x) 0 [1 2 3 4 5];  # => 15

  # Handle nulls safely
  version = from-maybe "0.0.0" (attrs.version or null);
in
  ...
```
