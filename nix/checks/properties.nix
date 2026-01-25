# nix/checks/properties.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                            // properties //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'Neuromancer,' the boy said, slitting long gray eyes against the
#      glare of the hologram. 'Neuro from the nerves, the silver paths.
#      Romancer. Necromancer. I call up the dead.'
#
#                                                         — Neuromancer
#
# Property-based testing framework for the prelude. Inspired by QuickCheck,
# adapted for Nix's evaluation model.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   properties.check       run all property tests
#   properties.for-all     generate test cases and verify property
#   properties.generators  sample data generators
#   properties.laws        algebraic law checkers
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib, pkgs }:
let
  # ─────────────────────────────────────────────────────────────────────────
  # Lisp-case aliases for lib.* and builtins.* functions
  # ─────────────────────────────────────────────────────────────────────────
  concat-strings-sep = lib.concatStringsSep;
  concat-map = lib.concatMap;
  map-attrs = lib.mapAttrs;
  to-string = builtins.toString;
  filter-attrs = lib.filterAttrs;
  map-attrs' = lib.mapAttrs';
  name-value-pair = lib.nameValuePair;
  to-upper = lib.toUpper;

  inherit (pkgs.aleph) run-command;

  list-filter = builtins.filter;
  list-length = builtins.length;
  try-eval = builtins.tryEval;
  deep-seq = builtins.deepSeq;
  type-of = builtins.typeOf;
  attr-names = builtins.attrNames;
  replace-strings = builtins.replaceStrings;

  # ─────────────────────────────────────────────────────────────────────────
  # Generators
  # ─────────────────────────────────────────────────────────────────────────
  # Static sample sets for property testing. Nix doesn't have randomness,
  # so we use carefully chosen edge cases and representative values.

  generators = {
    # Integers: edge cases and representative values
    int = [
      0
      1
      (-1)
      2
      (-2)
      10
      (-10)
      100
      (-100)
      999999
      (-999999)
    ];

    # Positive integers only
    positive-int = [
      1
      2
      3
      5
      10
      42
      100
      1000
    ];

    # Non-negative integers
    non-negative-int = [
      0
      1
      2
      3
      5
      10
      42
      100
    ];

    # Small integers for chunk sizes, indices, etc.
    small-int = [
      0
      1
      2
      3
      4
      5
      (-1)
      (-2)
    ];

    # Lists of various sizes and contents
    list-int = [
      [ ]
      [ 1 ]
      [
        1
        2
      ]
      [
        1
        2
        3
      ]
      [
        1
        1
        1
      ]
      [
        3
        1
        4
        1
        5
        9
      ]
      [
        (-1)
        0
        1
      ]
      [
        999
        (-999)
        0
      ]
    ];

    list-string = [
      [ ]
      [ "a" ]
      [
        "a"
        "b"
      ]
      [
        "hello"
        "world"
      ]
      [
        ""
        "a"
        ""
      ]
      [
        "foo"
        "bar"
        "baz"
      ]
    ];

    # Strings
    string = [
      ""
      "a"
      "ab"
      "hello"
      "hello world"
      "with\nnewline"
      "   spaces   "
    ];

    # Nullable values
    nullable = [
      null
      0
      1
      ""
      "value"
      [ ]
      { }
    ];

    # Either values
    either = [
      {
        _tag = "left";
        value = "error";
      }
      {
        _tag = "right";
        value = 42;
      }
      {
        _tag = "left";
        value = 0;
      }
      {
        _tag = "right";
        value = null;
      }
      {
        _tag = "right";
        value = [ ];
      }
    ];

    # Malformed values (for robustness testing)
    malformed-either = [
      { }
      { _tag = "left"; }
      { value = 42; }
      { _tag = "invalid"; }
      "not an attrset"
      null
      [ ]
    ];

    # Attribute sets
    attrs = [
      { }
      { a = 1; }
      {
        a = 1;
        b = 2;
      }
      {
        nested = {
          deep = 42;
        };
      }
    ];
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Property Combinators
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Test a property for all values in a generator.

    Returns { pass, failures } where failures is list of failing inputs.

    # Type

    ```
    for-all :: [a] -> (a -> Bool) -> { pass :: Bool; failures :: [a]; }
    ```
  */
  for-all =
    samples: prop:
    let
      results = map (x: {
        input = x;
        passed = prop x;
      }) samples;
      failures = map (r: r.input) (list-filter (r: !r.passed) results);
    in
    {
      pass = failures == [ ];
      inherit failures;
      total = list-length samples;
      passed = list-length samples - list-length failures;
    };

  /**
    Test a property for all pairs from two generators.

    # Type

    ```
    for-all2 :: [a] -> [b] -> (a -> b -> Bool) -> { pass :: Bool; failures :: [{ fst :: a; snd :: b }]; }
    ```
  */
  for-all2 =
    xs: ys: prop:
    let
      pairs = concat-map (
        x:
        map (y: {
          fst = x;
          snd = y;
        }) ys
      ) xs;
      results = map (p: {
        input = p;
        passed = prop p.fst p.snd;
      }) pairs;
      failures = map (r: r.input) (list-filter (r: !r.passed) results);
    in
    {
      pass = failures == [ ];
      inherit failures;
      total = list-length pairs;
      passed = list-length pairs - list-length failures;
    };

  # ─────────────────────────────────────────────────────────────────────────
  # Property Helpers
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Check that a function never throws (is total).

    Uses builtins.tryEval to catch evaluation errors.
  */
  terminates = f: x: (try-eval (deep-seq (f x) true)).success;

  /**
    Check that a function returns a specific type.
  */
  returns-type =
    type: f: x:
    type-of (f x) == type;

  /**
    Check that a function is idempotent: f(f(x)) == f(x)
  */
  idempotent = f: x: f (f x) == f x;

  # ─────────────────────────────────────────────────────────────────────────
  # Algebraic Laws
  # ─────────────────────────────────────────────────────────────────────────

  laws = {
    /**
      Identity law: f(id, x) == x
    */
    identity =
      f: id: samples:
      for-all samples (x: f id x == x);

    /**
      Associativity: f(f(a, b), c) == f(a, f(b, c))
    */
    associative =
      f: samples:
      let
        triples = concat-map (
          a:
          concat-map (
            b:
            map (c: {
              inherit a b c;
            }) samples
          ) samples
        ) samples;
      in
      for-all triples (t: f (f t.a t.b) t.c == f t.a (f t.b t.c));

    /**
      Commutativity: f(a, b) == f(b, a)
    */
    commutative = f: samples: for-all2 samples samples (a: b: f a b == f b a);

    /**
      Distributivity: f(a, g(b, c)) == g(f(a, b), f(a, c))
    */
    distributive =
      f: g: samples:
      let
        triples = concat-map (
          a:
          concat-map (
            b:
            map (c: {
              inherit a b c;
            }) samples
          ) samples
        ) samples;
      in
      for-all triples (t: f t.a (g t.b t.c) == g (f t.a t.b) (f t.a t.c));
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Prelude-Specific Properties
  # ─────────────────────────────────────────────────────────────────────────

  prelude-properties = prelude: {
    # ─────────────────────────────────────────────────────────────────────
    # Totality: functions don't crash on edge cases
    # ─────────────────────────────────────────────────────────────────────

    "safe.head terminates on all lists" = for-all generators.list-int (
      xs: terminates prelude.safe.head xs
    );

    "safe.tail terminates on all lists" = for-all generators.list-int (
      xs: terminates prelude.safe.tail xs
    );

    "safe.minimum terminates on all lists" = for-all generators.list-int (
      xs: terminates prelude.safe.minimum xs
    );

    "safe.maximum terminates on all lists" = for-all generators.list-int (
      xs: terminates prelude.safe.maximum xs
    );

    "safe.chunks-of terminates for all n" = for-all2 generators.small-int generators.list-int (
      n: xs: terminates (prelude.safe.chunks-of n) xs
    );

    "safe.take terminates for negative n" = for-all generators.list-int (
      xs: terminates (prelude.safe.take (-5)) xs
    );

    "safe.drop terminates for negative n" = for-all generators.list-int (
      xs: terminates (prelude.safe.drop (-5)) xs
    );

    "safe.elem-at terminates for any index" = for-all2 generators.small-int generators.list-int (
      i: xs: terminates (prelude.safe.elem-at xs) i
    );

    "safe.div terminates for zero divisor" = for-all generators.int (
      n: terminates (prelude.safe.div n) 0
    );

    "safe.is-left handles malformed input" = for-all generators.malformed-either (
      x: terminates prelude.safe.is-left x
    );

    "safe.is-right handles malformed input" = for-all generators.malformed-either (
      x: terminates prelude.safe.is-right x
    );

    # ─────────────────────────────────────────────────────────────────────
    # Correctness: functions return expected values
    # ─────────────────────────────────────────────────────────────────────

    "safe.head [] == null" = {
      pass = prelude.safe.head [ ] == null;
      failures = [ ];
    };

    "safe.minimum [] == null" = {
      pass = prelude.safe.minimum [ ] == null;
      failures = [ ];
    };

    "safe.maximum [] == null" = {
      pass = prelude.safe.maximum [ ] == null;
      failures = [ ];
    };

    "safe.chunks-of 0 xs == []" = for-all generators.list-int (xs: prelude.safe.chunks-of 0 xs == [ ]);

    "safe.take negative == []" = for-all generators.list-int (xs: prelude.safe.take (-1) xs == [ ]);

    "safe.drop negative == identity" = for-all generators.list-int (
      xs: prelude.safe.drop (-1) xs == xs
    );

    "safe.div x 0 == null" = for-all generators.int (x: prelude.safe.div x 0 == null);

    # ─────────────────────────────────────────────────────────────────────
    # Schema validation
    # ─────────────────────────────────────────────────────────────────────

    "schemas.either accepts valid Either" = for-all generators.either (e: prelude.schemas.either e);

    "schemas.either rejects malformed" = for-all generators.malformed-either (
      x: !prelude.schemas.either x
    );

    "schemas.positive-int accepts positive" = for-all generators.positive-int (
      n: prelude.schemas.positive-int n
    );

    "schemas.positive-int rejects zero" = {
      pass = !prelude.schemas.positive-int 0;
      failures = [ ];
    };

    "schemas.positive-int rejects negative" = for-all [
      (-1)
      (-10)
      (-999)
    ] (n: !prelude.schemas.positive-int n);
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Test Runner
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Run all properties and return results.
  */
  run-properties =
    prelude:
    let
      props = prelude-properties prelude;
      results = map-attrs (
        name: result:
        result
        // {
          inherit name;
        }
      ) props;
      failures = filter-attrs (_: r: !r.pass) results;
      passed = filter-attrs (_: r: r.pass) results;
    in
    {
      all = results;
      inherit failures passed;
      summary = {
        total = list-length (attr-names results);
        passed = list-length (attr-names passed);
        failed = list-length (attr-names failures);
        pass = failures == { };
      };
    };

  /**
    Format test results as a report.
  */
  format-report =
    results:
    let
      header = ''
        ════════════════════════════════════════════════════════════════════════════════
                                     PROPERTY TESTS
        ════════════════════════════════════════════════════════════════════════════════
      '';

      passed-section =
        if results.summary.passed > 0 then
          ''

            PASSED (${to-string results.summary.passed}):
            ${concat-strings-sep "\n" (map (name: "  [+] ${name}") (attr-names results.passed))}
          ''
        else
          "";

      failed-section =
        if results.summary.failed > 0 then
          ''

            FAILED (${to-string results.summary.failed}):
            ${concat-strings-sep "\n" (
              map (name: "  [-] ${name}: ${to-string (list-length results.failures.${name}.failures)} failures") (
                attr-names results.failures
              )
            )}
          ''
        else
          "";

      summary = ''

        ────────────────────────────────────────────────────────────────────────────────
        Total: ${to-string results.summary.total} | Passed: ${to-string results.summary.passed} | Failed: ${to-string results.summary.failed}
        ${if results.summary.pass then "ALL TESTS PASSED" else "SOME TESTS FAILED"}
        ────────────────────────────────────────────────────────────────────────────────
      '';
    in
    header + passed-section + failed-section + summary;

  # Render Dhall template with env vars (converts attr names to UPPER_SNAKE_CASE)
  render-dhall =
    name: src: vars:
    let
      env-vars = map-attrs' (
        k: v: name-value-pair (to-upper (replace-strings [ "-" ] [ "_" ] k)) (builtins.toString v)
      ) vars;
    in
    run-command name
      (
        {
          native-build-inputs = [ pkgs.haskellPackages.dhall ];
        }
        // env-vars
      )
      ''
        dhall text --file ${src} > $out
      '';

in
{
  inherit
    generators
    for-all
    for-all2
    terminates
    returns-type
    idempotent
    laws
    prelude-properties
    run-properties
    format-report
    ;

  /**
    Create a check derivation that runs property tests.
  */
  mk-check =
    prelude:
    let
      results = run-properties prelude;
      script = render-dhall "test-properties-script" ./scripts/test-properties.dhall {
        report = format-report results;
        result-message =
          if results.summary.pass then "All property tests passed." else "Property tests failed!";
        touch-out = if results.summary.pass then "touch $out" else "exit 1";
      };
    in
    run-command "prelude-properties" { } ''
      bash ${script}
    '';
}
