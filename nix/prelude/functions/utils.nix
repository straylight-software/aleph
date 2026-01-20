# nix/prelude/functions/utils.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // utils //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     He closed his eyes. Found the ridged face of the power stud.
#     And in the bloodlit dark behind his eyes, silver phosphenes
#     boiled in from the edge of space, hypnagogic images jerking
#     past like film compiled from random frames.
#
#                                                         — Neuromancer
#
# Debug, conditionals, paths, serialization, assertions. The utilities.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
rec {
  # ─────────────────────────────────────────────────────────────────────────
  # Debug
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Print a value to stderr during evaluation, then return a second value.

    The trace message is printed as a side effect during Nix evaluation.
    Useful for debugging lazy evaluation and understanding evaluation order.

    # Type

    ```
    trace :: String -> a -> a
    ```

    # Arguments

    - msg: message to print to stderr
    - val: value to return unchanged

    # Examples

    ```nix
    trace "computing x" (1 + 2)
    => 3  # prints "trace: computing x" to stderr
    ```
  */
  inherit (builtins) trace;

  /**
    Trace a value and return it (identity with side effect).

    Prints the value to stderr and returns the same value.
    Convenient for inspecting intermediate values in pipelines.

    # Type

    ```
    trace-val :: a -> a
    ```

    # Arguments

    - x: value to trace and return

    # Examples

    ```nix
    trace-val 42
    => 42  # prints "trace: 42" to stderr

    map trace-val [ 1 2 3 ]
    => [ 1 2 3 ]  # prints each element
    ```
  */
  trace-val = x: builtins.trace x x;

  /**
    Trace a value (forcing evaluation), then return a different value.

    Forces evaluation of the first argument via seq before returning
    the second. Useful for debugging when you want to trace one thing
    but return another.

    # Type

    ```
    trace-seq :: a -> b -> b
    ```

    # Arguments

    - x: value to trace (will be forced)
    - y: value to return

    # Examples

    ```nix
    trace-seq "debug info" (1 + 2)
    => 3  # prints "trace: debug info" to stderr

    trace-seq { a = 1; } "result"
    => "result"  # prints the attrset first
    ```
  */
  trace-seq = x: y: builtins.seq (builtins.trace x x) y;

  /**
    Trace a value with a descriptive label, showing JSON representation.

    Prefixes the trace output with a message and converts the value
    to JSON for readable output. Returns the original value unchanged.

    # Type

    ```
    trace-id :: String -> a -> a
    ```

    # Arguments

    - msg: descriptive label for the trace output
    - x: value to trace and return

    # Examples

    ```nix
    trace-id "config" { port = 8080; }
    => { port = 8080; }  # prints: trace: config: {"port":8080}

    trace-id "input" [ 1 2 3 ]
    => [ 1 2 3 ]  # prints: trace: input: [1,2,3]
    ```
  */
  trace-id = msg: x: builtins.trace "${msg}: ${builtins.toJSON x}" x;

  # ─────────────────────────────────────────────────────────────────────────
  # Conditionals
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Conditionally return an attribute set or empty.

    Returns the value if condition is true, otherwise an empty attribute set.
    Useful for conditionally including attributes in a set with the // operator.

    # Type

    ```
    when :: Bool -> AttrSet -> AttrSet
    ```

    # Arguments

    - cond: condition to check
    - val: attribute set to return if condition is true

    # Examples

    ```nix
    when true { foo = 1; }
    => { foo = 1; }

    when false { foo = 1; }
    => { }

    { bar = 2; } // when true { foo = 1; }
    => { bar = 2; foo = 1; }
    ```
  */
  when = cond: val: if cond then val else { };

  /**
    Alias for `when`. Conditionally return an attribute set or empty.

    Returns the value if condition is true, otherwise an empty attribute set.
    This alias makes the intent clearer when working with attribute sets.

    # Type

    ```
    when-attr :: Bool -> AttrSet -> AttrSet
    ```

    # Arguments

    - cond: condition to check
    - val: attribute set to return if condition is true

    # Examples

    ```nix
    when-attr true { foo = 1; }
    => { foo = 1; }

    when-attr false { foo = 1; }
    => { }

    { base = 1; } // when-attr (version > 2) { newFeature = true; }
    => { base = 1; }  # if version <= 2
    ```
  */
  when-attr = when;

  /**
    Conditionally wrap a value in a singleton list or return empty.

    Returns a single-element list containing the value if condition is true,
    otherwise an empty list. Useful for conditionally including items in lists.

    # Type

    ```
    when-list :: Bool -> a -> [a]
    ```

    # Arguments

    - cond: condition to check
    - val: value to wrap in a list if condition is true

    # Examples

    ```nix
    when-list true "item"
    => [ "item" ]

    when-list false "item"
    => [ ]

    [ "always" ] ++ when-list enableFeature "optional"
    => [ "always" "optional" ]  # if enableFeature is true
    ```
  */
  when-list = cond: val: if cond then [ val ] else [ ];

  /**
    Conditionally return a list or empty list.

    Returns the list if condition is true, otherwise an empty list.
    Unlike `when-list`, this takes a list directly instead of wrapping a value.

    # Type

    ```
    when-lists :: Bool -> [a] -> [a]
    ```

    # Arguments

    - cond: condition to check
    - val: list to return if condition is true

    # Examples

    ```nix
    when-lists true [ 1 2 3 ]
    => [ 1 2 3 ]

    when-lists false [ 1 2 3 ]
    => [ ]

    [ "base" ] ++ when-lists enableExtras [ "extra1" "extra2" ]
    => [ "base" ]  # if enableExtras is false
    ```
  */
  when-lists = cond: val: if cond then val else [ ];

  /**
    Conditionally return a string or empty string.

    Returns the string if condition is true, otherwise an empty string.
    Useful for conditionally including text in string concatenation.

    # Type

    ```
    when-str :: Bool -> String -> String
    ```

    # Arguments

    - cond: condition to check
    - val: string to return if condition is true

    # Examples

    ```nix
    when-str true "hello"
    => "hello"

    when-str false "hello"
    => ""

    "prefix" + when-str verbose " --verbose" + " suffix"
    => "prefix --verbose suffix"  # if verbose is true
    ```
  */
  when-str = cond: val: if cond then val else "";

  # ─────────────────────────────────────────────────────────────────────────
  # Paths
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Get the directory portion of a path.

    Returns the parent directory of the given path, stripping the
    final component (file or directory name).

    # Type

    ```
    dirname :: Path -> Path
    ```

    # Arguments

    - path: path to extract directory from

    # Examples

    ```nix
    dirname /foo/bar/baz.txt
    => /foo/bar

    dirname /foo/bar/
    => /foo/bar

    dirname ./src/main.nix
    => ./src
    ```
  */
  dirname = builtins.dirOf;

  /**
    Get the final component of a path (file or directory name).

    Returns the last component of a path as a string, stripping
    the directory portion.

    # Type

    ```
    basename :: Path -> String
    ```

    # Arguments

    - path: path to extract basename from

    # Examples

    ```nix
    basename /foo/bar/baz.txt
    => "baz.txt"

    basename /foo/bar
    => "bar"

    basename ./src/main.nix
    => "main.nix"
    ```
  */
  basename = builtins.baseNameOf;

  /**
    Check if a path exists in the filesystem.

    Returns true if the path exists (file, directory, or symlink),
    false otherwise. Works during Nix evaluation time.

    # Type

    ```
    path-exists :: Path -> Bool
    ```

    # Arguments

    - path: path to check for existence

    # Examples

    ```nix
    path-exists /etc/passwd
    => true

    path-exists /nonexistent/path
    => false

    path-exists ./flake.nix
    => true  # if flake.nix exists in current directory
    ```
  */
  path-exists = builtins.pathExists;

  /**
    Read the contents of a file as a string.

    Returns the entire contents of the file as a string.
    The file must exist and be readable at evaluation time.

    # Type

    ```
    read-file :: Path -> String
    ```

    # Arguments

    - path: path to the file to read

    # Examples

    ```nix
    read-file ./version.txt
    => "1.0.0\n"

    read-file /etc/hostname
    => "myhost\n"
    ```
  */
  read-file = builtins.readFile;

  /**
    List directory contents as an attribute set.

    Returns an attribute set where keys are entry names and values
    are entry types: "regular", "directory", "symlink", or "unknown".

    # Type

    ```
    read-dir :: Path -> AttrSet String
    ```

    # Arguments

    - path: path to the directory to read

    # Examples

    ```nix
    read-dir ./src
    => { "main.nix" = "regular"; "lib" = "directory"; }

    read-dir /tmp
    => { "file.txt" = "regular"; "link" = "symlink"; ... }
    ```
  */
  read-dir = builtins.readDir;

  /**
    Convert a string to a path.

    Converts a string representation of a path to an actual path value.
    Note: This function is deprecated in newer Nix versions.

    # Type

    ```
    to-path :: String -> Path
    ```

    # Arguments

    - str: string to convert to a path

    # Examples

    ```nix
    to-path "/foo/bar"
    => /foo/bar

    to-path "./relative"
    => /current/working/dir/relative
    ```
  */
  to-path = builtins.toPath;

  # ─────────────────────────────────────────────────────────────────────────
  # Serialization
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Convert a Nix value to a JSON string.

    Serializes a Nix value (attribute set, list, string, number, bool, null)
    to its JSON representation as a string.

    # Type

    ```
    to-json :: a -> String
    ```

    # Arguments

    - value: Nix value to serialize

    # Examples

    ```nix
    to-json { foo = 1; bar = [ 2 3 ]; }
    => "{\"bar\":[2,3],\"foo\":1}"

    to-json [ 1 2 3 ]
    => "[1,2,3]"

    to-json null
    => "null"
    ```
  */
  to-json = builtins.toJSON;

  /**
    Parse a JSON string into a Nix value.

    Deserializes a JSON string into the corresponding Nix value
    (attribute set, list, string, number, bool, null).

    # Type

    ```
    from-json :: String -> a
    ```

    # Arguments

    - str: JSON string to parse

    # Examples

    ```nix
    from-json "{\"foo\": 1, \"bar\": [2, 3]}"
    => { foo = 1; bar = [ 2 3 ]; }

    from-json "[1, 2, 3]"
    => [ 1 2 3 ]

    from-json "null"
    => null
    ```
  */
  from-json = builtins.fromJSON;

  /**
    Convert an attribute set to a TOML string.

    Serializes a Nix attribute set to TOML format. Only supports
    attribute sets at the top level (not lists or primitives).

    # Type

    ```
    to-toml :: AttrSet -> String
    ```

    # Arguments

    - attrs: attribute set to serialize

    # Examples

    ```nix
    to-toml { server.port = 8080; server.host = "localhost"; }
    => "[server]\nhost = \"localhost\"\nport = 8080\n"

    to-toml { name = "myapp"; version = "1.0"; }
    => "name = \"myapp\"\nversion = \"1.0\"\n"
    ```
  */
  to-toml = lib.generators.toTOML { };

  # ─────────────────────────────────────────────────────────────────────────
  # Assertions
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Assert a condition with a custom error message.

    Returns the value if the condition is true, otherwise throws
    an error with the provided message. Useful for validating inputs.

    # Type

    ```
    assert-msg :: Bool -> String -> a -> a
    ```

    # Arguments

    - cond: condition that must be true
    - msg: error message to throw if condition is false
    - val: value to return if condition is true

    # Examples

    ```nix
    assert-msg true "never shown" 42
    => 42

    assert-msg false "value must be positive" 42
    => error: value must be positive

    assert-msg (x > 0) "x must be positive" x
    => x  # if x > 0
    ```
  */
  assert-msg =
    cond: msg: val:
    if cond then val else builtins.throw msg;

  /**
    Throw an error with a message, aborting evaluation.

    Immediately aborts Nix evaluation with the given error message.
    The return type is "never" (!) indicating this function does not return.

    # Type

    ```
    throw :: String -> !
    ```

    # Arguments

    - msg: error message to display

    # Examples

    ```nix
    throw "something went wrong"
    => error: something went wrong

    if x == null then throw "x is required" else x
    => x  # if x is not null
    ```
  */
  inherit (builtins) throw;

  /**
    Attempt to evaluate a value, catching errors.

    Returns an attribute set with `success` (bool) and `value` fields.
    If evaluation succeeds, `success` is true and `value` contains the result.
    If evaluation fails, `success` is false and `value` is false.

    # Type

    ```
    try-eval :: a -> { success :: Bool; value :: a | Bool; }
    ```

    # Arguments

    - expr: expression to attempt to evaluate

    # Examples

    ```nix
    try-eval (1 + 1)
    => { success = true; value = 2; }

    try-eval (throw "error")
    => { success = false; value = false; }

    try-eval (builtins.readFile "/nonexistent")
    => { success = false; value = false; }
    ```
  */
  try-eval = builtins.tryEval;

}
