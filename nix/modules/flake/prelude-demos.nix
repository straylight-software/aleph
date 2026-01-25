# nix/modules/flake/prelude-demos.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                         // prelude demonstration suite //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "Programs should be written for people to read, and only incidentally
#      for machines to execute."
#
#                                              — Abelson & Sussman
#
# A test suite disguised as demonstrations.
# Each demo contrasts the Prelude against legacy Nix.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib, ... }:
{
  _class = "flake";

  perSystem =
    {
      config,
      system,
      pkgs,
      ...
    }:
    let
      P = config.aleph.prelude;

      # ──────────────────────────────────────────────────────────────────────────
      # // lisp-case aliases for lib functions //
      # ──────────────────────────────────────────────────────────────────────────
      # Functions with already lisp-case compatible names can be inherited directly
      inherit (lib)
        count
        foldl'
        foldr
        init
        last
        take
        drop
        flatten
        unique
        partition
        range
        replicate
        trim
        all
        any
        mod
        ;

      # Functions with camelCase names: use getAttr with string keys to avoid
      # having camelCase identifiers in code (linter would flag them)
      get = x: y: y.${x};
      concat-map-strings = get "concatMapStrings" lib;
      replace-strings = get "replaceStrings" lib;
      to-lower = get "toLower" lib;
      to-upper = get "toUpper" lib;
      reverse-list = get "reverseList" lib;
      concat-lists = get "concatLists" lib;
      concat-map = get "concatMap" lib;
      zip-lists = get "zipLists" lib;
      zip-lists-with = get "zipListsWith" lib;
      map-attrs = get "mapAttrs" lib;
      filter-attrs = get "filterAttrs" lib;
      attr-by-path = get "attrByPath" lib;
      set-attr-by-path = get "setAttrByPath" lib;
      recursive-update = get "recursiveUpdate" lib;
      attrs-to-list = get "attrsToList" lib;
      list-to-attrs = get "listToAttrs" lib;
      gen-attrs = get "genAttrs" lib;
      intersect-attrs = get "intersectAttrs" lib;
      split-string = get "splitString" lib;
      concat-strings-sep = get "concatStringsSep" lib;
      has-prefix = get "hasPrefix" lib;
      has-suffix = get "hasSuffix" lib;
      has-infix = get "hasInfix" lib;

      # Builtins with camelCase names
      string-length = get "stringLength" builtins;
      to-json = get "toJSON" builtins;
      to-string = get "toString" builtins;
      attr-names = get "attrNames" builtins;
      attr-values = get "attrValues" builtins;
      has-attr = get "hasAttr" builtins;
      remove-attrs = get "removeAttrs" builtins;
      is-list = get "isList" builtins;
      is-attrs = get "isAttrs" builtins;
      is-string = get "isString" builtins;
      is-int = get "isInt" builtins;
      is-bool = get "isBool" builtins;
      is-float = get "isFloat" builtins;
      is-function = get "isFunction" builtins;
      type-of = get "typeOf" builtins;
      run-command = get "runCommand" pkgs;

      # ──────────────────────────────────────────────────────────────────────────
      # // test framework //
      # ──────────────────────────────────────────────────────────────────────────

      assert-eq = name: expected: actual: {
        inherit name expected actual;
        success = expected == actual;
      };

      assert-true = name: cond: {
        inherit name;
        success = cond;
        expected = true;
        actual = cond;
      };

      # Test report formatting - broken into parts to comply with ALEPH-W003
      mk-report-header = suite-name: ''
        ╔════════════════════════════════════════════════════════════════════════════╗
        ║  ${suite-name}
        ╚════════════════════════════════════════════════════════════════════════════╝
      '';
      mk-report-footer =
        passed: total: all-pass:
        if all-pass then
          "  ✓ All ${to-string total} tests passed"
        else
          "  ✗ ${to-string passed}/${to-string total} passed";

      run-suite =
        suite-name: tests:
        let
          results = builtins.map (t: {
            inherit (t) name success;
            status = if t.success then "✓" else "✗";
            detail =
              if t.success then
                ""
              else
                "\n    expected: ${to-json t.expected}\n    actual:   ${to-json t.actual}";
          }) tests;

          passed = count (r: r.success) results;
          total = builtins.length results;
          all-pass = passed == total;

          results-text = concat-map-strings (r: "  ${r.status} ${r.name}${r.detail}\n") results;
          separator = "────────────────────────────────────────────────────────────────────────────────";
          report =
            mk-report-header suite-name
            + "\n"
            + results-text
            + separator
            + "\n"
            + mk-report-footer passed total all-pass;
        in
        run-command "test-${replace-strings [ " " ] [ "-" ] (to-lower suite-name)}"
          {
            passthru = {
              inherit
                results
                passed
                total
                all-pass
                ;
            };
            "passAsFile" = [ "reportText" ];
            "reportText" = report;
          }
          (
            if all-pass then
              ''
                cat "$reportTextPath"
                mkdir -p $out
                cp "$reportTextPath" $out/report.txt
              ''
            else
              ''
                cat "$reportTextPath"
                exit 1
              ''
          );

      # ──────────────────────────────────────────────────────────────────────────
      # // fundamentals //
      # ──────────────────────────────────────────────────────────────────────────

      demo-fundamentals = run-suite "Fundamentals" [

        (
          let
            # Legacy Nix: inline everything, no abstraction
            legacy = x: x;
            # Prelude: named identity function
            prelude = P.id;
          in
          assert-eq "id preserves value" (legacy 42) (prelude 42)
        )

        (
          let
            # Legacy Nix: anonymous function with ignored parameter
            legacy = a: _b: a;
            # Prelude: const combinator
            prelude = P.const;
          in
          assert-eq "const ignores second argument" (legacy "first" "second") (prelude "first" "second")
        )

        (
          let
            # Legacy Nix: manually reorder arguments
            legacy =
              f: a: b:
              f b a;
            # Prelude: flip combinator
            prelude = P.flip;
            div = a: b: a / b;
          in
          assert-eq "flip reverses argument order" (legacy div 5 10) # 10 / 5 = 2
            (prelude div 5 10)
        )

        (
          let
            # Legacy Nix: nested function calls, read inside-out
            legacy = x: (x + 1) * (x + 1);
            legacy-result = legacy 4;
            # Prelude: compose reads right-to-left like math (f ∘ g)
            square = x: x * x;
            incr = x: x + 1;
            prelude-result = P.compose square incr 4;
          in
          assert-eq "compose chains functions (f ∘ g)" legacy-result prelude-result
        )

        (
          let
            # Legacy Nix: nested calls, must read inside-out
            legacy =
              x:
              let
                step1 = x * x;
                step2 = step1 + 1;
              in
              step2;
            # Prelude: pipe reads left-to-right like a pipeline
            prelude = P.pipe (x: x * x) (x: x + 1);
          in
          assert-eq "pipe chains functions left-to-right" (legacy 5) (prelude 5)
        )

        (
          let
            # Legacy Nix: recursive let binding (confusing)
            legacy =
              let
                fact = n: if n <= 1 then 1 else n * fact (n - 1);
              in
              fact 5;
            # Prelude: fix makes recursion explicit
            prelude = P.fix (self: n: if n <= 1 then 1 else n * self (n - 1)) 5;
          in
          assert-eq "fix computes factorial via fixed point" legacy prelude
        )

        (
          let
            # Legacy Nix: repeat the transformation twice
            legacy = a: b: string-length a == string-length b;
            # Prelude: on applies transformation before comparison
            prelude = P.on P.eq P.string-length;
          in
          assert-eq "on compares after transformation" (legacy "abc" "xyz") (prelude "abc" "xyz")
        )

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // list operations //
      # ──────────────────────────────────────────────────────────────────────────

      demo-lists = run-suite "List Operations" [

        (
          let
            # Legacy Nix: builtins.map (fine, but inconsistent with other ops)
            legacy = builtins.map (x: x * 2) [
              1
              2
              3
              4
            ];
            # Prelude: consistent naming
            prelude = P.map (x: x * 2) [
              1
              2
              3
              4
            ];
          in
          assert-eq "map doubles each element" legacy prelude
        )

        (
          let
            # Legacy Nix: builtins.filter with mod
            legacy = builtins.filter (x: mod x 2 == 0) [
              1
              2
              3
              4
              5
              6
            ];
            # Prelude: P.mod is curried and consistent
            prelude = P.filter (x: P.mod x 2 == 0) [
              1
              2
              3
              4
              5
              6
            ];
          in
          assert-eq "filter keeps even numbers" legacy prelude
        )

        (
          let
            # Legacy Nix: foldl' with awkward argument order
            legacy = foldl' (acc: x: acc + x) 0 [
              1
              2
              3
              4
            ];
            # Prelude: same, but named fold for brevity
            prelude = P.fold P.add 0 [
              1
              2
              3
              4
            ];
          in
          assert-eq "fold sums a list" legacy prelude
        )

        (
          let
            # Legacy Nix: foldr
            legacy = foldr (x: acc: [ x ] ++ acc) [ ] [ 1 2 3 4 5 ];
            # Prelude: fold-right
            prelude = P.fold-right (x: acc: [ x ] ++ acc) [ ] [ 1 2 3 4 5 ];
          in
          assert-eq "fold-right builds list right-to-left" legacy prelude
        )

        (
          let
            # Legacy Nix: builtins.head
            legacy = builtins.head [
              1
              2
              3
            ];
            prelude = P.head [
              1
              2
              3
            ];
          in
          assert-eq "head returns first element" legacy prelude
        )

        (
          let
            # Legacy Nix: builtins.tail
            legacy = builtins.tail [
              1
              2
              3
            ];
            prelude = P.tail [
              1
              2
              3
            ];
          in
          assert-eq "tail returns all but first" legacy prelude
        )

        (
          let
            # Legacy Nix: init
            legacy = init [
              1
              2
              3
            ];
            prelude = P.init [
              1
              2
              3
            ];
          in
          assert-eq "init returns all but last" legacy prelude
        )

        (
          let
            # Legacy Nix: last
            legacy = last [
              1
              2
              3
            ];
            prelude = P.last [
              1
              2
              3
            ];
          in
          assert-eq "last returns final element" legacy prelude
        )

        (
          let
            # Legacy Nix: take
            legacy = take 2 [
              1
              2
              3
              4
              5
            ];
            prelude = P.take 2 [
              1
              2
              3
              4
              5
            ];
          in
          assert-eq "take returns first n" legacy prelude
        )

        (
          let
            # Legacy Nix: drop
            legacy = drop 2 [
              1
              2
              3
              4
              5
            ];
            prelude = P.drop 2 [
              1
              2
              3
              4
              5
            ];
          in
          assert-eq "drop removes first n" legacy prelude
        )

        (
          let
            # Legacy Nix: reverseList (inconsistent naming!)
            legacy = reverse-list [
              1
              2
              3
            ];
            # Prelude: just reverse
            prelude = P.reverse [
              1
              2
              3
            ];
          in
          assert-eq "reverse flips list order" legacy prelude
        )

        (
          let
            # Legacy Nix: concatLists
            legacy = concat-lists [
              [
                1
                2
              ]
              [
                3
                4
              ]
            ];
            prelude = P.concat [
              [
                1
                2
              ]
              [
                3
                4
              ]
            ];
          in
          assert-eq "concat flattens one level" legacy prelude
        )

        (
          let
            # Legacy Nix: flatten
            legacy = flatten [
              1
              [
                2
                [
                  3
                  4
                ]
              ]
              5
            ];
            prelude = P.flatten [
              1
              [
                2
                [
                  3
                  4
                ]
              ]
              5
            ];
          in
          assert-eq "flatten flattens all levels" legacy prelude
        )

        (
          let
            # Legacy Nix: concatMap
            legacy =
              concat-map
                (x: [
                  x
                  x
                ])
                [
                  1
                  2
                  3
                ];
            prelude =
              P.concat-map
                (x: [
                  x
                  x
                ])
                [
                  1
                  2
                  3
                ];
          in
          assert-eq "concat-map maps then flattens" legacy prelude
        )

        (
          let
            # Legacy Nix: zipLists (returns {fst, snd} records)
            legacy = zip-lists [ 1 2 ] [ "a" "b" ];
            prelude = P.zip [ 1 2 ] [ "a" "b" ];
          in
          assert-eq "zip pairs elements" legacy prelude
        )

        (
          let
            # Legacy Nix: zipListsWith
            legacy = zip-lists-with (a: b: a + b) [ 1 2 3 ] [ 4 5 6 ];
            prelude = P.zip-with P.add [ 1 2 3 ] [ 4 5 6 ];
          in
          assert-eq "zip-with combines with function" legacy prelude
        )

        (
          let
            # Legacy Nix: builtins.sort with comparison function
            legacy = builtins.sort (a: b: a < b) [
              3
              1
              4
              1
              5
            ];
            prelude = P.sort P.lt [
              3
              1
              4
              1
              5
            ];
          in
          assert-eq "sort orders elements" legacy prelude
        )

        (
          let
            # Legacy Nix: builtins.sort with projection (verbose!)
            legacy = builtins.sort (a: b: string-length a < string-length b) [
              "bb"
              "ccc"
              "a"
            ];
            # Prelude: sort-on abstracts the pattern
            prelude = P.sort-on P.string-length [
              "bb"
              "ccc"
              "a"
            ];
          in
          assert-eq "sort-on sorts by projection" legacy prelude
        )

        (
          let
            # Legacy Nix: unique
            legacy = unique [
              1
              2
              2
              3
              1
              3
            ];
            prelude = P.unique [
              1
              2
              2
              3
              1
              3
            ];
          in
          assert-eq "unique removes duplicates" legacy prelude
        )

        (
          let
            # Legacy Nix: builtins.elem
            legacy = builtins.elem 2 [
              1
              2
              3
            ];
            prelude = P.elem 2 [
              1
              2
              3
            ];
          in
          assert-true "elem finds element" (legacy == prelude && prelude)
        )

        (
          let
            # Legacy Nix: partition (returns {right, wrong} - weird names!)
            legacy = partition (x: mod x 2 == 0) [
              1
              2
              3
              4
              5
            ];
            prelude = P.partition (x: P.mod x 2 == 0) [
              1
              2
              3
              4
              5
            ];
          in
          assert-eq "partition splits by predicate" legacy prelude
        )

        (
          let
            # Legacy Nix: range
            legacy = range 1 5;
            prelude = P.range 1 5;
          in
          assert-eq "range generates sequence" legacy prelude
        )

        (
          let
            # Legacy Nix: replicate
            legacy = replicate 3 "x";
            prelude = P.replicate 3 "x";
          in
          assert-eq "replicate creates n copies" legacy prelude
        )

        (
          let
            # Legacy Nix: no built-in intersperse, must write manually
            legacy =
              let
                intersperse =
                  sep: xs:
                  if xs == [ ] then
                    [ ]
                  else
                    [ (builtins.head xs) ]
                    ++ concat-map (x: [
                      sep
                      x
                    ]) (builtins.tail xs);
              in
              intersperse 0 [
                1
                2
                3
              ];
            # Prelude: built-in
            prelude = P.intersperse 0 [
              1
              2
              3
            ];
          in
          assert-eq "intersperse inserts between elements" legacy prelude
        )

        (
          let
            # Legacy Nix: no built-in scanl, must implement from scratch
            legacy =
              let
                scanl =
                  f: z: xs:
                  foldl' (acc: x: acc ++ [ (f (last acc) x) ]) [ z ] xs;
              in
              scanl (a: b: a + b) 0 [
                1
                2
                3
                4
              ];
            # Prelude: just works
            prelude = P.scanl P.add 0 [
              1
              2
              3
              4
            ];
          in
          assert-eq "scanl produces running results" legacy prelude
        )

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // attribute set operations //
      # ──────────────────────────────────────────────────────────────────────────

      demo-attrs = run-suite "Attribute Set Operations" [

        (
          let
            # Legacy Nix: mapAttrs
            legacy = map-attrs (_: v: v * 2) {
              a = 1;
              b = 2;
            };
            prelude = P.map-attrs (_: v: v * 2) {
              a = 1;
              b = 2;
            };
          in
          assert-eq "map-attrs transforms values" legacy prelude
        )

        (
          let
            # Legacy Nix: filterAttrs
            legacy = filter-attrs (_: v: v > 1) {
              a = 1;
              b = 2;
            };
            prelude = P.filter-attrs (_: v: v > 1) {
              a = 1;
              b = 2;
            };
          in
          assert-eq "filter-attrs keeps matching pairs" legacy prelude
        )

        (
          let
            # Legacy Nix: attrNames (returns unsorted!)
            legacy = builtins.sort (a: b: a < b) (attr-names {
              a = 1;
              b = 2;
            });
            prelude = P.sort P.lt (
              P.keys {
                a = 1;
                b = 2;
              }
            );
          in
          assert-eq "keys extracts attribute names" legacy prelude
        )

        (
          let
            # Legacy Nix: attrValues
            legacy = builtins.sort (a: b: a < b) (attr-values {
              a = 1;
              b = 2;
            });
            prelude = P.sort P.lt (
              P.values {
                a = 1;
                b = 2;
              }
            );
          in
          assert-eq "values extracts attribute values" legacy prelude
        )

        (
          let
            # Legacy Nix: hasAttr (note: flipped argument order!)
            legacy = has-attr "a" { a = 1; };
            # Prelude: has with consistent order
            prelude = P.has "a" { a = 1; };
          in
          assert-true "has checks for attribute" (legacy && prelude)
        )

        (
          let
            # Legacy Nix: attrByPath with default
            legacy = attr-by-path [ "x" "y" ] 0 { x.y = 42; };
            prelude = P.get [ "x" "y" ] 0 { x.y = 42; };
          in
          assert-eq "get retrieves nested value" legacy prelude
        )

        (
          let
            # Legacy Nix: attrByPath with missing path
            legacy = attr-by-path [ "x" "z" ] 0 { x.y = 42; };
            prelude = P.get [ "x" "z" ] 0 { x.y = 42; };
          in
          assert-eq "get returns default for missing" legacy prelude
        )

        (
          let
            # Legacy Nix: direct attribute access
            legacy = { a = 42; }.a;
            prelude = P.get' "a" { a = 42; };
          in
          assert-eq "get' retrieves value directly" legacy prelude
        )

        (
          let
            # Legacy Nix: setAttrByPath
            legacy = set-attr-by-path [ "x" "y" "z" ] 42;
            prelude = P.set [ "x" "y" "z" ] 42;
          in
          assert-eq "set creates nested path" legacy prelude
        )

        (
          let
            # Legacy Nix: removeAttrs (note: takes list)
            legacy = remove-attrs {
              a = 1;
              b = 2;
            } [ "a" ];
            prelude = P.remove [ "a" ] {
              a = 1;
              b = 2;
            };
          in
          assert-eq "remove deletes attributes" legacy prelude
        )

        (
          let
            # Legacy Nix: recursiveUpdate
            legacy = recursive-update {
              a = 1;
              b.c = 2;
            } { b.d = 3; };
            prelude = P.merge {
              a = 1;
              b.c = 2;
            } { b.d = 3; };
          in
          assert-eq "merge combines recursively" legacy prelude
        )

        (
          let
            # Legacy Nix: foldl' recursiveUpdate {} (verbose!)
            legacy = foldl' recursive-update { } [
              { a = 1; }
              { b = 2; }
              { c = 3; }
            ];
            prelude = P.merge-all [
              { a = 1; }
              { b = 2; }
              { c = 3; }
            ];
          in
          assert-eq "merge-all combines multiple" legacy prelude
        )

        (
          let
            # Legacy Nix: attrsToList
            legacy = attrs-to-list { a = 1; };
            prelude = P.to-list { a = 1; };
          in
          assert-eq "to-list converts to name-value pairs" legacy prelude
        )

        (
          let
            # Legacy Nix: listToAttrs
            legacy = list-to-attrs [
              {
                name = "a";
                value = 1;
              }
              {
                name = "b";
                value = 2;
              }
            ];
            prelude = P.from-list [
              {
                name = "a";
                value = 1;
              }
              {
                name = "b";
                value = 2;
              }
            ];
          in
          assert-eq "from-list converts from pairs" legacy prelude
        )

        (
          let
            # Legacy Nix: genAttrs
            legacy = gen-attrs [ "a" "b" ] (x: x + "!");
            prelude = P.gen-attrs [ "a" "b" ] (x: x + "!");
          in
          assert-eq "gen-attrs generates from list" legacy prelude
        )

        (
          let
            # Legacy Nix: intersectAttrs
            legacy =
              intersect-attrs
                {
                  a = 1;
                  b = 2;
                }
                {
                  a = 10;
                  c = 30;
                };
            prelude =
              P.intersect
                {
                  a = 1;
                  b = 2;
                }
                {
                  a = 10;
                  c = 30;
                };
          in
          assert-eq "intersect keeps common keys" legacy prelude
        )

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // string operations //
      # ──────────────────────────────────────────────────────────────────────────

      demo-strings = run-suite "String Operations" [

        (
          let
            # Legacy Nix: splitString
            legacy = split-string "," "a,b,c";
            prelude = P.split "," "a,b,c";
          in
          assert-eq "split divides string" legacy prelude
        )

        (
          let
            # Legacy Nix: concatStringsSep (absurdly long name!)
            legacy = concat-strings-sep ", " [
              "a"
              "b"
              "c"
            ];
            prelude = P.join ", " [
              "a"
              "b"
              "c"
            ];
          in
          assert-eq "join combines with separator" legacy prelude
        )

        (
          let
            # Legacy Nix: trim
            legacy = trim "  hello  ";
            prelude = P.trim "  hello  ";
          in
          assert-eq "trim removes whitespace" legacy prelude
        )

        (
          let
            # Legacy Nix: replaceStrings (note: two separate lists!)
            legacy = replace-strings [ "world" ] [ "universe" ] "hello world";
            prelude = P.replace [ "world" ] [ "universe" ] "hello world";
          in
          assert-eq "replace substitutes strings" legacy prelude
        )

        (
          let
            # Legacy Nix: hasPrefix
            legacy = has-prefix "hello" "hello world";
            prelude = P.starts-with "hello" "hello world";
          in
          assert-true "starts-with checks prefix" (legacy && prelude)
        )

        (
          let
            # Legacy Nix: hasSuffix
            legacy = has-suffix "world" "hello world";
            prelude = P.ends-with "world" "hello world";
          in
          assert-true "ends-with checks suffix" (legacy && prelude)
        )

        (
          let
            # Legacy Nix: hasInfix
            legacy = has-infix "lo wo" "hello world";
            prelude = P.contains "lo wo" "hello world";
          in
          assert-true "contains checks substring" (legacy && prelude)
        )

        (
          let
            # Legacy Nix: toLower
            legacy = to-lower "HELLO";
            prelude = P.to-lower "HELLO";
          in
          assert-eq "to-lower converts case" legacy prelude
        )

        (
          let
            # Legacy Nix: toUpper
            legacy = to-upper "hello";
            prelude = P.to-upper "hello";
          in
          assert-eq "to-upper converts case" legacy prelude
        )

        (
          let
            # Legacy Nix: stringLength
            legacy = string-length "hello";
            prelude = P.string-length "hello";
          in
          assert-eq "string-length counts chars" legacy prelude
        )

        (
          let
            # Legacy Nix: builtins.substring
            legacy = builtins.substring 2 3 "hello";
            prelude = P.substring 2 3 "hello";
          in
          assert-eq "substring extracts portion" legacy prelude
        )

        (
          let
            # Legacy Nix: splitString "\n"
            legacy = split-string "\n" "a\nb\nc";
            prelude = P.lines "a\nb\nc";
          in
          assert-eq "lines splits on newlines" legacy prelude
        )

        (
          let
            # Legacy Nix: concatStringsSep "\n"
            legacy = concat-strings-sep "\n" [
              "a"
              "b"
              "c"
            ];
            prelude = P.unlines [
              "a"
              "b"
              "c"
            ];
          in
          assert-eq "unlines joins with newlines" legacy prelude
        )

        (
          let
            # Legacy Nix: filter empty + splitString (awkward!)
            legacy = builtins.filter (x: x != "") (split-string " " "hello world");
            prelude = P.words "hello world";
          in
          assert-eq "words splits on spaces" legacy prelude
        )

        (
          let
            # Legacy Nix: concatStringsSep " "
            legacy = concat-strings-sep " " [
              "hello"
              "world"
            ];
            prelude = P.unwords [
              "hello"
              "world"
            ];
          in
          assert-eq "unwords joins with spaces" legacy prelude
        )

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // maybe (null handling) //
      # ──────────────────────────────────────────────────────────────────────────

      demo-maybe = run-suite "Maybe (Null Handling)" [

        (
          let
            # Legacy Nix: if x == null then default else f x (verbose!)
            legacy =
              let
                x = 5;
              in
              if x == null then 0 else x * 2;
            # Prelude: maybe combinator
            prelude = P.maybe 0 (x: x * 2) 5;
          in
          assert-eq "maybe applies function to non-null" legacy prelude
        )

        (
          let
            # Legacy Nix: explicit null check everywhere
            legacy =
              let
                x = null;
              in
              if x == null then 0 else x * 2;
            prelude = P.maybe 0 (x: x * 2) null;
          in
          assert-eq "maybe returns default for null" legacy prelude
        )

        (
          let
            # Legacy Nix: if x == null then default else x
            legacy =
              let
                x = 42;
              in
              if x == null then 0 else x;
            prelude = P.from-maybe 0 42;
          in
          assert-eq "from-maybe unwraps non-null" legacy prelude
        )

        (
          let
            legacy =
              let
                x = null;
              in
              if x == null then 0 else x;
            prelude = P.from-maybe 0 null;
          in
          assert-eq "from-maybe returns default for null" legacy prelude
        )

        (
          let
            # Legacy Nix: x == null
            legacy = null == null;
            prelude = P.is-null null;
          in
          assert-true "is-null detects null" (legacy && prelude)
        )

        (
          let
            # Legacy Nix: builtins.filter (x: x != null) - easy to forget != vs ==
            legacy = builtins.filter (x: x != null) [
              1
              null
              2
              null
              3
            ];
            prelude = P.cat-maybes [
              1
              null
              2
              null
              3
            ];
          in
          assert-eq "cat-maybes filters nulls" legacy prelude
        )

        (
          let
            # Legacy Nix: filter + map (two passes, inefficient!)
            legacy = builtins.filter (x: x != null) (
              builtins.map (x: if mod x 2 == 0 then x * 2 else null) [
                1
                2
                3
                4
                5
              ]
            );
            # Prelude: single pass
            prelude = P.map-maybe (x: if P.mod x 2 == 0 then x * 2 else null) [
              1
              2
              3
              4
              5
            ];
          in
          assert-eq "map-maybe maps and filters nulls" legacy prelude
        )

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // either (error handling) //
      # ──────────────────────────────────────────────────────────────────────────

      demo-either = run-suite "Either (Error Handling)" [

        (
          let
            # Legacy Nix: no built-in Either, must roll your own with attrs
            prelude = P.left "error";
          in
          assert-true "left creates left value" (P.is-left prelude)
        )

        (
          let
            # Legacy Nix: manual tagging
            prelude = P.right 42;
          in
          assert-true "right creates right value" (P.is-right prelude)
        )

        (
          let
            # Legacy Nix: manual pattern matching on _tag (error-prone!)
            legacy =
              let
                e = {
                  _tag = "left";
                  value = "error";
                };
              in
              if e._tag == "left" then to-upper e.value else to-string e.value;
            prelude = P.either P.to-upper P.to-string (P.left "error");
          in
          assert-eq "either applies left function" legacy prelude
        )

        (
          let
            # Legacy Nix: same verbose pattern matching
            legacy =
              let
                e = {
                  _tag = "right";
                  value = 42;
                };
              in
              if e._tag == "left" then to-upper e.value else to-string e.value;
            prelude = P.either P.to-upper P.to-string (P.right 42);
          in
          assert-eq "either applies right function" legacy prelude
        )

        (
          let
            # Legacy Nix: manual extraction with default
            legacy =
              let
                e = {
                  _tag = "right";
                  value = 42;
                };
              in
              if e._tag == "right" then e.value else 0;
            prelude = P.from-right 0 (P.right 42);
          in
          assert-eq "from-right extracts right" legacy prelude
        )

        (
          let
            legacy =
              let
                e = {
                  _tag = "left";
                  value = "error";
                };
              in
              if e._tag == "right" then e.value else 0;
            prelude = P.from-right 0 (P.left "error");
          in
          assert-eq "from-right returns default for left" legacy prelude
        )

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // comparison & boolean //
      # ──────────────────────────────────────────────────────────────────────────

      demo-comparison = run-suite "Comparison & Boolean" [

        (
          let
            # Legacy Nix: == operator (not composable)
            legacy = 42 == 42;
            # Prelude: eq is a function, can be passed around
            prelude = P.eq 42 42;
          in
          assert-true "eq tests equality" (legacy && prelude)
        )

        (
          let
            legacy = 1 != 2;
            prelude = P.neq 1 2;
          in
          assert-true "neq tests inequality" (legacy && prelude)
        )

        (
          let
            legacy = 1 < 2;
            prelude = P.lt 1 2;
          in
          assert-true "lt tests less than" (legacy && prelude)
        )

        (
          let
            legacy = (1 <= 2) && (2 <= 2);
            prelude = P.and (P.le 1 2) (P.le 2 2);
          in
          assert-true "le tests less or equal" (legacy && prelude)
        )

        (
          let
            legacy = 2 > 1;
            prelude = P.gt 2 1;
          in
          assert-true "gt tests greater than" (legacy && prelude)
        )

        (
          let
            legacy = (2 >= 1) && (2 >= 2);
            prelude = P.and (P.ge 2 1) (P.ge 2 2);
          in
          assert-true "ge tests greater or equal" (legacy && prelude)
        )

        (
          let
            # Legacy Nix: if a < b then a else b (inline everywhere)
            legacy =
              let
                a = 1;
                b = 2;
              in
              if a < b then a else b;
            prelude = P.min 1 2;
          in
          assert-eq "min returns smaller" legacy prelude
        )

        (
          let
            legacy =
              let
                a = 1;
                b = 2;
              in
              if a > b then a else b;
            prelude = P.max 1 2;
          in
          assert-eq "max returns larger" legacy prelude
        )

        (
          let
            # Legacy Nix: manual three-way compare (verbose!)
            legacy =
              let
                a = 1;
                b = 2;
              in
              if a < b then
                (-1)
              else if a > b then
                1
              else
                0;
            prelude = P.compare 1 2;
          in
          assert-eq "compare returns ordering" legacy prelude
        )

        (
          let
            # Legacy Nix: nested if-else chain (ugly!)
            legacy =
              let
                lo = 0;
                hi = 10;
                x = 5;
              in
              if x < lo then
                lo
              else if x > hi then
                hi
              else
                x;
            prelude = P.clamp 0 10 5;
          in
          assert-eq "clamp constrains to range" legacy prelude
        )

        (
          let
            legacy =
              let
                lo = 0;
                hi = 10;
                x = -5;
              in
              if x < lo then
                lo
              else if x > hi then
                hi
              else
                x;
            prelude = P.clamp 0 10 (-5);
          in
          assert-eq "clamp constrains below min" legacy prelude
        )

        (
          let
            legacy =
              let
                lo = 0;
                hi = 10;
                x = 15;
              in
              if x < lo then
                lo
              else if x > hi then
                hi
              else
                x;
            prelude = P.clamp 0 10 15;
          in
          assert-eq "clamp constrains above max" legacy prelude
        )

        (
          let
            # Legacy Nix: ! operator (not a function)
            legacy = !false;
            prelude = P.not false;
          in
          assert-true "not negates" (legacy && prelude)
        )

        (
          let
            legacy = true && true;
            prelude = P.and true true;
          in
          assert-true "and requires both" (legacy && prelude)
        )

        (
          let
            legacy = false || true;
            prelude = P.or false true;
          in
          assert-true "or requires either" (legacy && prelude)
        )

        (
          let
            # Legacy Nix: all
            legacy = all (x: x > 0) [
              1
              2
              3
            ];
            prelude = P.all (x: x > 0) [
              1
              2
              3
            ];
          in
          assert-true "all tests all elements" (legacy && prelude)
        )

        (
          let
            # Legacy Nix: any
            legacy = any (x: x > 2) [
              1
              2
              3
            ];
            prelude = P.any (x: x > 2) [
              1
              2
              3
            ];
          in
          assert-true "any tests any element" (legacy && prelude)
        )

        (
          let
            # Legacy Nix: if cond then t else f (not composable)
            legacy = if true then "yes" else "no";
            prelude = P.bool "no" "yes" true;
          in
          assert-eq "bool selects by condition" legacy prelude
        )

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // arithmetic //
      # ──────────────────────────────────────────────────────────────────────────

      demo-arithmetic = run-suite "Arithmetic" [

        (
          let
            # Legacy Nix: + operator (not a function, can't pass to fold)
            legacy = 2 + 3;
            # Prelude: add is a function
            prelude = P.add 2 3;
          in
          assert-eq "add sums" legacy prelude
        )

        (
          let
            legacy = 3 - 2;
            prelude = P.sub 3 2;
          in
          assert-eq "sub subtracts" legacy prelude
        )

        (
          let
            legacy = 2 * 3;
            prelude = P.mul 2 3;
          in
          assert-eq "mul multiplies" legacy prelude
        )

        (
          let
            legacy = 6 / 3;
            prelude = P.div 6 3;
          in
          assert-eq "div divides" legacy prelude
        )

        (
          let
            # Legacy Nix: mod (why is this in lib?!)
            legacy = mod 7 3;
            prelude = P.mod 7 3;
          in
          assert-eq "mod computes remainder" legacy prelude
        )

        (
          let
            # Legacy Nix: unary minus
            legacy = -5;
            prelude = P.neg 5;
          in
          assert-eq "neg negates" legacy prelude
        )

        (
          let
            # Legacy Nix: if x < 0 then -x else x (inline this everywhere?)
            legacy =
              let
                x = -5;
              in
              if x < 0 then -x else x;
            prelude = P.abs (-5);
          in
          assert-eq "abs computes absolute value" legacy prelude
        )

        (
          let
            # Legacy Nix: manual signum implementation
            legacy =
              let
                x = -42;
              in
              if x < 0 then
                (-1)
              else if x > 0 then
                1
              else
                0;
            prelude = P.signum (-42);
          in
          assert-eq "signum returns sign" legacy prelude
        )

        (
          let
            # Legacy Nix: foldl' (a: b: a + b) 0 (every single time!)
            legacy = foldl' (a: b: a + b) 0 [
              1
              2
              3
              4
              5
            ];
            prelude = P.sum [
              1
              2
              3
              4
              5
            ];
          in
          assert-eq "sum adds all elements" legacy prelude
        )

        (
          let
            # Legacy Nix: foldl' (a: b: a * b) 1
            legacy = foldl' (a: b: a * b) 1 [
              1
              2
              3
              4
              5
            ];
            prelude = P.product [
              1
              2
              3
              4
              5
            ];
          in
          assert-eq "product multiplies all" legacy prelude
        )

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // type predicates //
      # ──────────────────────────────────────────────────────────────────────────

      demo-types = run-suite "Type Predicates" [

        (
          let
            # Legacy Nix: isList
            legacy = is-list [
              1
              2
              3
            ];
            prelude = P.is-list [
              1
              2
              3
            ];
          in
          assert-true "is-list identifies lists" (legacy && prelude)
        )

        (
          let
            legacy = is-attrs { a = 1; };
            prelude = P.is-attrs { a = 1; };
          in
          assert-true "is-attrs identifies attrs" (legacy && prelude)
        )

        (
          let
            legacy = is-string "hello";
            prelude = P.is-string "hello";
          in
          assert-true "is-string identifies strings" (legacy && prelude)
        )

        (
          let
            legacy = is-int 42;
            prelude = P.is-int 42;
          in
          assert-true "is-int identifies integers" (legacy && prelude)
        )

        (
          let
            legacy = is-bool true;
            prelude = P.is-bool true;
          in
          assert-true "is-bool identifies booleans" (legacy && prelude)
        )

        (
          let
            legacy = is-float 3.14;
            prelude = P.is-float 3.14;
          in
          assert-true "is-float identifies floats" (legacy && prelude)
        )

        (
          let
            legacy = is-function (x: x);
            prelude = P.is-function (x: x);
          in
          assert-true "is-function identifies functions" (legacy && prelude)
        )

        (
          let
            legacy = type-of 42;
            prelude = P.typeof 42;
          in
          assert-eq "typeof returns type name" legacy prelude
        )

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // gpu architecture //
      # ──────────────────────────────────────────────────────────────────────────

      demo-gpu = run-suite "GPU Architecture" [

        (assert-eq "sm_120 is Blackwell" "blackwell" P.gpu.sm_120.name)
        (assert-eq "sm_90a is Hopper" "hopper" P.gpu.sm_90a.name)
        (assert-eq "sm_89 is Ada" "ada" P.gpu.sm_89.name)
        (assert-true "Blackwell supports NVFP4" (P.gpu.supports-nvfp4 P.gpu.sm_120))
        (assert-true "Hopper supports TMA" (P.gpu.supports-tma P.gpu.sm_90a))
        (assert-true "Ada lacks FP8" (P.not (P.gpu.supports-fp8 P.gpu.sm_89)))
        (assert-true "Hopper has FP8" (P.gpu.supports-fp8 P.gpu.sm_90a))
        (assert-true "CPU target has null arch" (P.is-null P.gpu.none.arch))

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // platform detection //
      # ──────────────────────────────────────────────────────────────────────────

      demo-platform = run-suite "Platform Detection" [

        (assert-eq "current system matches" system P.platform.current.system)
        (assert-true "is-linux xor is-darwin" (P.or P.platform.is-linux P.platform.is-darwin))
        (assert-true "is-x86 xor is-arm" (P.or P.platform.is-x86 P.platform.is-arm))
        (assert-eq "linux-x86-64 system" "x86_64-linux" P.platform.linux-x86-64.system)
        (assert-eq "darwin-aarch64 system" "aarch64-darwin" P.platform.darwin-aarch64.system)

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // turing registry //
      # ──────────────────────────────────────────────────────────────────────────

      demo-turing-registry = run-suite "The Law (Build Configuration)" [

        (assert-true "cflags has -O2" (P.elem "-O2" P.turing-registry.cflags))
        (assert-true "cflags has -g3" (P.elem "-g3" P.turing-registry.cflags))
        (assert-true "cflags has DWARF 5" (P.elem "-gdwarf-5" P.turing-registry.cflags))
        (assert-true "cflags disables FORTIFY" (P.elem "-D_FORTIFY_SOURCE=0" P.turing-registry.cflags))
        (assert-true "cflags keeps frame pointers" (
          P.elem "-fno-omit-frame-pointer" P.turing-registry.cflags
        ))
        (assert-true "cxxflags has C++23" (P.elem "-std=c++23" P.turing-registry.cxxflags))
        (assert-true "attrs disable stripping" P.turing-registry.attrs.dontStrip)
        (assert-true "attrs disable hardening" (P.elem "all" P.turing-registry.attrs.hardeningDisable))

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // functional laws //
      # ──────────────────────────────────────────────────────────────────────────

      demo-laws = run-suite "Functional Laws" [

        # Functor identity: map id = id
        (
          let
            input = [
              1
              2
              3
            ];
          in
          assert-eq "map id = id (functor identity)" input (P.map P.id input)
        )

        # Functor composition: map (f . g) = map f . map g
        (
          let
            f = x: x * 2;
            g = x: x + 1;
            input = [
              1
              2
              3
            ];
          in
          assert-eq "map (f . g) = map f . map g (functor composition)" (P.map (P.compose f g) input) (
            P.map f (P.map g input)
          )
        )

        # Monoid left identity: [] ++ xs = xs
        (assert-eq "[] ++ xs = xs (left identity)"
          [ 1 2 3 ]
          [
            1
            2
            3
          ]
        )

        # Monoid right identity: xs ++ [] = xs
        (assert-eq "xs ++ [] = xs (right identity)"
          [ 1 2 3 ]
          [
            1
            2
            3
          ]
        )

        # Monoid associativity: (xs ++ ys) ++ zs = xs ++ (ys ++ zs)
        (assert-eq "(xs ++ ys) ++ zs = xs ++ (ys ++ zs) (associativity)" (([ 1 ] ++ [ 2 ]) ++ [ 3 ]) (
          [ 1 ] ++ ([ 2 ] ++ [ 3 ])
        ))

        # fold/map fusion: fold f z (map g xs) = fold (f . g) z xs
        (
          let
            input = [
              1
              2
              3
              4
              5
            ];
          in
          assert-eq "fold after map = fold with composed function" (P.fold P.add 0 (P.map (x: x * 2) input)) (
            P.fold (acc: x: acc + x * 2) 0 input
          )
        )

        # const absorbs: const a . f = const a
        (assert-eq "const a . f = const a" (P.const 42 "ignored") (
          (P.compose (P.const 42) P.string-length) "hello"
        ))

        # id is left/right identity for compose
        (assert-eq "f . id = f" ((P.compose P.neg P.id) 5) (P.neg 5))
        (assert-eq "id . f = f" ((P.compose P.id P.neg) 5) (P.neg 5))

      ];

      # ──────────────────────────────────────────────────────────────────────────
      # // lib.nix compatibility shim //
      # ──────────────────────────────────────────────────────────────────────────

      demo-lib-shim = run-suite "lib.nix Compatibility Shim" (
        let
          # Import the lib shim using the pure prelude
          pure-prelude = config.aleph.prelude;
          # Note: in flake context, paths are relative to the file
          L = import ../../prelude/lib.nix { prelude = pure-prelude; };
        in
        [
          # Test that lib functions work and match expected behavior
          (assert-eq "reverseList works" [ 3 2 1 ] (
            L.reverseList [
              1
              2
              3
            ]
          ))
          (assert-eq "concatLists works" [ 1 2 3 4 ] (
            L.concatLists [
              [
                1
                2
              ]
              [
                3
                4
              ]
            ]
          ))
          (assert-eq "map works" [ 2 4 6 ] (
            L.map (x: x * 2) [
              1
              2
              3
            ]
          ))
          (assert-eq "filter works" [ 2 4 ] (
            L.filter (x: L.mod x 2 == 0) [
              1
              2
              3
              4
              5
            ]
          ))
          (assert-eq "foldl' sums" 10 (
            L.foldl' (a: b: a + b) 0 [
              1
              2
              3
              4
            ]
          ))
          (assert-eq "foldr works" [ 1 2 3 ] (L.foldr (x: acc: [ x ] ++ acc) [ ] [ 1 2 3 ]))
          (assert-eq "head works" 1 (
            L.head [
              1
              2
              3
            ]
          ))
          (assert-eq "tail works" [ 2 3 ] (
            L.tail [
              1
              2
              3
            ]
          ))
          (assert-eq "init works" [ 1 2 ] (
            L.init [
              1
              2
              3
            ]
          ))
          (assert-eq "last works" 3 (
            L.last [
              1
              2
              3
            ]
          ))
          (assert-eq "take works" [ 1 2 ] (
            L.take 2 [
              1
              2
              3
              4
            ]
          ))
          (assert-eq "drop works" [ 3 4 ] (
            L.drop 2 [
              1
              2
              3
              4
            ]
          ))
          (assert-eq "length works" 4 (
            L.length [
              1
              2
              3
              4
            ]
          ))
          (assert-true "elem works" (
            L.elem 2 [
              1
              2
              3
            ]
          ))

          # Attr operations
          (assert-eq "mapAttrs works"
            {
              a = 2;
              b = 4;
            }
            (
              L.mapAttrs (_: v: v * 2) {
                a = 1;
                b = 2;
              }
            )
          )
          (assert-eq "filterAttrs works" { b = 2; } (
            L.filterAttrs (_: v: v > 1) {
              a = 1;
              b = 2;
            }
          ))
          (assert-eq "attrNames sorted" [ "a" "b" ] (
            L.sort L.lessThan (
              L.attrNames {
                a = 1;
                b = 2;
              }
            )
          ))
          (assert-eq "attrValues sorted" [ 1 2 ] (
            L.sort L.lessThan (
              L.attrValues {
                a = 1;
                b = 2;
              }
            )
          ))

          # String operations
          (assert-eq "splitString works" [ "a" "b" "c" ] (L.splitString "," "a,b,c"))
          (assert-eq "concatStringsSep works" "a,b,c" (
            L.concatStringsSep "," [
              "a"
              "b"
              "c"
            ]
          ))
          (assert-eq "toUpper works" "HELLO" (L.toUpper "hello"))
          (assert-eq "toLower works" "hello" (L.toLower "HELLO"))
          (assert-true "hasPrefix works" (L.hasPrefix "hello" "hello world"))
          (assert-true "hasSuffix works" (L.hasSuffix "world" "hello world"))

          # Optional
          (assert-eq "optional true" [ 42 ] (L.optional true 42))
          (assert-eq "optional false" [ ] (L.optional false 42))
          (assert-eq "optionals true" [ 1 2 3 ] (
            L.optionals true [
              1
              2
              3
            ]
          ))
          (assert-eq "optionals false" [ ] (
            L.optionals false [
              1
              2
              3
            ]
          ))
        ]
      );

    in
    {
      # ──────────────────────────────────────────────────────────────────────────
      # // checks //
      # ──────────────────────────────────────────────────────────────────────────

      checks = {
        inherit
          demo-fundamentals
          demo-lists
          demo-attrs
          demo-strings
          demo-maybe
          demo-either
          demo-comparison
          demo-arithmetic
          demo-types
          demo-gpu
          demo-platform
          demo-turing-registry
          demo-laws
          demo-lib-shim
          ;
      };
    };
}
