# nix/prelude/functions/strings.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                               // strings //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'Case.' He felt her crouch beside him, close. 'We gotta be
#      going.' But he could not bring himself to turn, to look at
#      her. His vision crawled with ghosts, translucent, trembling,
#      super-imposed on the grid of the Kuang program.
#
#                                                         — Neuromancer
#
# String operations. Concatenation, splitting, matching, transformation.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
rec {
  # ─────────────────────────────────────────────────────────────────────────
  # Strings
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Split a string by a separator into a list of substrings.

    # Type

    ```
    split :: String -> String -> [String]
    ```

    # Arguments

    - sep: separator string to split on
    - str: string to split

    # Examples

    ```nix
    split "/" "a/b/c"
    => [ "a" "b" "c" ]
    ```
  */
  split = lib.splitString;

  /**
    Join a list of strings with a separator.

    # Type

    ```
    join :: String -> [String] -> String
    ```

    # Arguments

    - sep: separator to insert between elements
    - strs: list of strings to join

    # Examples

    ```nix
    join ", " [ "a" "b" "c" ]
    => "a, b, c"
    ```
  */
  join = lib.concatStringsSep;

  /**
    Remove leading and trailing whitespace from a string.

    # Type

    ```
    trim :: String -> String
    ```

    # Arguments

    - str: string to trim

    # Examples

    ```nix
    trim "  hello world  "
    => "hello world"
    ```
  */
  inherit (lib) trim;

  /**
    Replace all occurrences of substrings in a string.

    # Type

    ```
    replace :: [String] -> [String] -> String -> String
    ```

    # Arguments

    - from: list of substrings to replace
    - to: list of replacement strings (same length as from)
    - str: string to perform replacements on

    # Examples

    ```nix
    replace [ "foo" "bar" ] [ "FOO" "BAR" ] "foo and bar"
    => "FOO and BAR"
    ```
  */
  replace = lib.replaceStrings;

  /**
    Check if a string starts with a given prefix.

    # Type

    ```
    starts-with :: String -> String -> Bool
    ```

    # Arguments

    - prefix: prefix to check for
    - str: string to check

    # Examples

    ```nix
    starts-with "hello" "hello world"
    => true
    ```
  */
  starts-with = lib.hasPrefix;

  /**
    Check if a string ends with a given suffix.

    # Type

    ```
    ends-with :: String -> String -> Bool
    ```

    # Arguments

    - suffix: suffix to check for
    - str: string to check

    # Examples

    ```nix
    ends-with ".nix" "default.nix"
    => true
    ```
  */
  ends-with = lib.hasSuffix;

  /**
    Check if a string contains a given substring.

    # Type

    ```
    contains :: String -> String -> Bool
    ```

    # Arguments

    - infix: substring to search for
    - str: string to search in

    # Examples

    ```nix
    contains "world" "hello world"
    => true
    ```
  */
  contains = lib.hasInfix;

  /**
    Convert a string to lowercase.

    # Type

    ```
    to-lower :: String -> String
    ```

    # Arguments

    - str: string to convert

    # Examples

    ```nix
    to-lower "Hello World"
    => "hello world"
    ```
  */
  to-lower = lib.toLower;

  /**
    Convert a string to uppercase.

    # Type

    ```
    to-upper :: String -> String
    ```

    # Arguments

    - str: string to convert

    # Examples

    ```nix
    to-upper "Hello World"
    => "HELLO WORLD"
    ```
  */
  to-upper = lib.toUpper;

  /**
    Convert any value to its string representation.

    # Type

    ```
    to-string :: a -> String
    ```

    # Arguments

    - x: value to convert

    # Examples

    ```nix
    to-string 42
    => "42"
    ```
  */
  to-string = builtins.toString;

  /**
    Return the length of a string in characters.

    # Type

    ```
    string-length :: String -> Int
    ```

    # Arguments

    - str: string to measure

    # Examples

    ```nix
    string-length "hello"
    => 5
    ```
  */
  string-length = builtins.stringLength;

  /**
    Extract a substring from a string.

    # Type

    ```
    substring :: Int -> Int -> String -> String
    ```

    # Arguments

    - start: starting index (0-based)
    - len: number of characters to extract (-1 for rest of string)
    - str: source string

    # Examples

    ```nix
    substring 0 5 "hello world"
    => "hello"
    ```
  */
  inherit (builtins) substring;

  /**
    Escape special characters in a string by prefixing with backslash.

    # Type

    ```
    escape :: [String] -> String -> String
    ```

    # Arguments

    - chars: list of characters to escape
    - str: string to escape

    # Examples

    ```nix
    escape [ "$" ] "cost: $100"
    => "cost: \\$100"
    ```
  */
  inherit (lib) escape;

  /**
    Escape a string for safe use as a shell argument.

    # Type

    ```
    escape-shell :: String -> String
    ```

    # Arguments

    - str: string to escape

    # Examples

    ```nix
    escape-shell "hello world"
    => "'hello world'"
    ```
  */
  escape-shell = lib.escapeShellArg;

  /**
    Split a string into lines.

    # Type

    ```
    lines :: String -> [String]
    ```

    # Arguments

    - str: string to split

    # Examples

    ```nix
    lines "a\nb\nc"
    => [ "a" "b" "c" ]
    ```
  */
  lines = s: lib.splitString "\n" s;

  /**
    Join a list of strings with newlines.

    # Type

    ```
    unlines :: [String] -> String
    ```

    # Arguments

    - strs: list of strings to join

    # Examples

    ```nix
    unlines [ "a" "b" "c" ]
    => "a\nb\nc"
    ```
  */
  unlines = lib.concatStringsSep "\n";

  /**
    Split a string into words (non-empty substrings separated by spaces).

    # Type

    ```
    words :: String -> [String]
    ```

    # Arguments

    - str: string to split

    # Examples

    ```nix
    words "hello   world"
    => [ "hello" "world" ]
    ```
  */
  words = s: builtins.filter (x: x != "") (lib.splitString " " s);

  /**
    Join a list of strings with spaces.

    # Type

    ```
    unwords :: [String] -> String
    ```

    # Arguments

    - strs: list of strings to join

    # Examples

    ```nix
    unwords [ "hello" "world" ]
    => "hello world"
    ```
  */
  unwords = lib.concatStringsSep " ";

  /**
    Capitalize the first character of a string.

    # Type

    ```
    capitalize :: String -> String
    ```

    # Arguments

    - str: string to capitalize

    # Examples

    ```nix
    capitalize "hello"
    => "Hello"
    ```
  */
  capitalize =
    s: if s == "" then "" else lib.toUpper (builtins.substring 0 1 s) + builtins.substring 1 (-1) s;

  /**
    Check if a string is blank (empty or contains only whitespace).

    # Type

    ```
    is-blank :: String -> Bool
    ```

    # Arguments

    - str: string to check

    # Examples

    ```nix
    is-blank "   "
    => true
    ```
  */
  is-blank = s: lib.trim s == "";

  /**
    Repeat a string n times.

    # Type

    ```
    repeat :: Int -> String -> String
    ```

    # Arguments

    - n: number of repetitions
    - str: string to repeat

    # Examples

    ```nix
    repeat 3 "ab"
    => "ababab"
    ```
  */
  repeat = n: s: lib.concatStrings (lib.replicate n s);

  /**
    Pad a string on the left to a minimum width.

    # Type

    ```
    pad-left :: Int -> String -> String -> String
    ```

    # Arguments

    - n: minimum width
    - char: character to pad with
    - str: string to pad

    # Examples

    ```nix
    pad-left 5 "0" "42"
    => "00042"
    ```
  */
  pad-left =
    n: char: s:
    let
      len = builtins.stringLength s;
      padding = if len >= n then "" else repeat (n - len) char;
    in
    padding + s;

  /**
    Pad a string on the right to a minimum width.

    # Type

    ```
    pad-right :: Int -> String -> String -> String
    ```

    # Arguments

    - n: minimum width
    - char: character to pad with
    - str: string to pad

    # Examples

    ```nix
    pad-right 5 "." "hi"
    => "hi..."
    ```
  */
  pad-right =
    n: char: s:
    let
      len = builtins.stringLength s;
      padding = if len >= n then "" else repeat (n - len) char;
    in
    s + padding;

}
