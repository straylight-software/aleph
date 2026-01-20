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
#   properties.forAll      generate test cases and verify property
#   properties.generators  sample data generators
#   properties.laws        algebraic law checkers
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib, pkgs }:
let
  inherit (lib) concatStringsSep;

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
    forAll :: [a] -> (a -> Bool) -> { pass :: Bool; failures :: [a]; }
    ```
  */
  forAll =
    samples: prop:
    let
      results = map (x: {
        input = x;
        passed = prop x;
      }) samples;
      failures = map (r: r.input) (builtins.filter (r: !r.passed) results);
    in
    {
      pass = failures == [ ];
      inherit failures;
      total = builtins.length samples;
      passed = builtins.length samples - builtins.length failures;
    };

  /**
    Test a property for all pairs from two generators.

    # Type

    ```
    forAll2 :: [a] -> [b] -> (a -> b -> Bool) -> { pass :: Bool; failures :: [{ fst :: a; snd :: b }]; }
    ```
  */
  forAll2 =
    xs: ys: prop:
    let
      pairs = lib.concatMap (
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
      failures = map (r: r.input) (builtins.filter (r: !r.passed) results);
    in
    {
      pass = failures == [ ];
      inherit failures;
      total = builtins.length pairs;
      passed = builtins.length pairs - builtins.length failures;
    };

  # ─────────────────────────────────────────────────────────────────────────
  # Property Helpers
  # ─────────────────────────────────────────────────────────────────────────

  /**
    Check that a function never throws (is total).

    Uses builtins.tryEval to catch evaluation errors.
  */
  terminates = f: x: (builtins.tryEval (builtins.deepSeq (f x) true)).success;

  /**
    Check that a function returns a specific type.
  */
  returns-type =
    type: f: x:
    builtins.typeOf (f x) == type;

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
      forAll samples (x: f id x == x);

    /**
      Associativity: f(f(a, b), c) == f(a, f(b, c))
    */
    associative =
      f: samples:
      let
        triples = lib.concatMap (
          a:
          lib.concatMap (
            b:
            map (c: {
              inherit a b c;
            }) samples
          ) samples
        ) samples;
      in
      forAll triples (t: f (f t.a t.b) t.c == f t.a (f t.b t.c));

    /**
      Commutativity: f(a, b) == f(b, a)
    */
    commutative = f: samples: forAll2 samples samples (a: b: f a b == f b a);

    /**
      Distributivity: f(a, g(b, c)) == g(f(a, b), f(a, c))
    */
    distributive =
      f: g: samples:
      let
        triples = lib.concatMap (
          a:
          lib.concatMap (
            b:
            map (c: {
              inherit a b c;
            }) samples
          ) samples
        ) samples;
      in
      forAll triples (t: f t.a (g t.b t.c) == g (f t.a t.b) (f t.a t.c));
  };

  # ─────────────────────────────────────────────────────────────────────────
  # Prelude-Specific Properties
  # ─────────────────────────────────────────────────────────────────────────

  preludeProperties = prelude: {
    # ─────────────────────────────────────────────────────────────────────
    # Totality: functions don't crash on edge cases
    # ─────────────────────────────────────────────────────────────────────

    "safe.head terminates on all lists" = forAll generators.list-int (
      xs: terminates prelude.safe.head xs
    );

    "safe.tail terminates on all lists" = forAll generators.list-int (
      xs: terminates prelude.safe.tail xs
    );

    "safe.minimum terminates on all lists" = forAll generators.list-int (
      xs: terminates prelude.safe.minimum xs
    );

    "safe.maximum terminates on all lists" = forAll generators.list-int (
      xs: terminates prelude.safe.maximum xs
    );

    "safe.chunks-of terminates for all n" = forAll2 generators.small-int generators.list-int (
      n: xs: terminates (prelude.safe.chunks-of n) xs
    );

    "safe.take terminates for negative n" = forAll generators.list-int (
      xs: terminates (prelude.safe.take (-5)) xs
    );

    "safe.drop terminates for negative n" = forAll generators.list-int (
      xs: terminates (prelude.safe.drop (-5)) xs
    );

    "safe.elem-at terminates for any index" = forAll2 generators.small-int generators.list-int (
      i: xs: terminates (prelude.safe.elem-at xs) i
    );

    "safe.div terminates for zero divisor" = forAll generators.int (
      n: terminates (prelude.safe.div n) 0
    );

    "safe.is-left handles malformed input" = forAll generators.malformed-either (
      x: terminates prelude.safe.is-left x
    );

    "safe.is-right handles malformed input" = forAll generators.malformed-either (
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

    "safe.chunks-of 0 xs == []" = forAll generators.list-int (xs: prelude.safe.chunks-of 0 xs == [ ]);

    "safe.take negative == []" = forAll generators.list-int (xs: prelude.safe.take (-1) xs == [ ]);

    "safe.drop negative == identity" = forAll generators.list-int (xs: prelude.safe.drop (-1) xs == xs);

    "safe.div x 0 == null" = forAll generators.int (x: prelude.safe.div x 0 == null);

    # ─────────────────────────────────────────────────────────────────────
    # Schema validation
    # ─────────────────────────────────────────────────────────────────────

    "schemas.either accepts valid Either" = forAll generators.either (e: prelude.schemas.either e);

    "schemas.either rejects malformed" = forAll generators.malformed-either (
      x: !prelude.schemas.either x
    );

    "schemas.positive-int accepts positive" = forAll generators.positive-int (
      n: prelude.schemas.positive-int n
    );

    "schemas.positive-int rejects zero" = {
      pass = !prelude.schemas.positive-int 0;
      failures = [ ];
    };

    "schemas.positive-int rejects negative" = forAll [
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
  runProperties =
    prelude:
    let
      props = preludeProperties prelude;
      results = lib.mapAttrs (
        name: result:
        result
        // {
          inherit name;
        }
      ) props;
      failures = lib.filterAttrs (_: r: !r.pass) results;
      passed = lib.filterAttrs (_: r: r.pass) results;
    in
    {
      all = results;
      inherit failures passed;
      summary = {
        total = builtins.length (builtins.attrNames results);
        passed = builtins.length (builtins.attrNames passed);
        failed = builtins.length (builtins.attrNames failures);
        pass = failures == { };
      };
    };

  /**
    Format test results as a report.
  */
  formatReport =
    results:
    let
      header = ''
        ════════════════════════════════════════════════════════════════════════════════
                                     PROPERTY TESTS
        ════════════════════════════════════════════════════════════════════════════════
      '';

      passedSection =
        if results.summary.passed > 0 then
          ''

            PASSED (${toString results.summary.passed}):
            ${concatStringsSep "\n" (map (name: "  [+] ${name}") (builtins.attrNames results.passed))}
          ''
        else
          "";

      failedSection =
        if results.summary.failed > 0 then
          ''

            FAILED (${toString results.summary.failed}):
            ${concatStringsSep "\n" (
              map (
                name: "  [-] ${name}: ${toString (builtins.length results.failures.${name}.failures)} failures"
              ) (builtins.attrNames results.failures)
            )}
          ''
        else
          "";

      summary = ''

        ────────────────────────────────────────────────────────────────────────────────
        Total: ${toString results.summary.total} | Passed: ${toString results.summary.passed} | Failed: ${toString results.summary.failed}
        ${if results.summary.pass then "ALL TESTS PASSED" else "SOME TESTS FAILED"}
        ────────────────────────────────────────────────────────────────────────────────
      '';
    in
    header + passedSection + failedSection + summary;

in
{
  inherit
    generators
    forAll
    forAll2
    terminates
    returns-type
    idempotent
    laws
    preludeProperties
    runProperties
    formatReport
    ;

  /**
    Create a check derivation that runs property tests.
  */
  mkCheck =
    prelude:
    pkgs.runCommand "prelude-properties" { } ''
      ${
        let
          results = runProperties prelude;
        in
        if results.summary.pass then
          ''
            echo "${formatReport results}"
            echo "All property tests passed."
            touch $out
          ''
        else
          ''
            echo "${formatReport results}"
            echo "Property tests failed!"
            exit 1
          ''
      }
    '';
}
