let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "long-inline-string"
    , language = "nix"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "indented_string_expression"
      , regex = Some "(?s)(?:.*\\n){12,}"
      }
    , message = "ALEPH-W003: long inline string"
    , note =
      { description =
          "A multi-line string exceeds 10 lines. Consider extracting to a separate file."
      , examples =
        [ ''
          write-shell-bin "my-script" '''
            set -euo pipefail

            foobar () {
              echo "hello world"

              exit 1
            }

            cat my_file.txt

            foobar
          '${"'"}''
        ]
      , suggested_fix =
          ''
          Move long scripts to separate files:

          ```nix
          # Instead of inline
          buildPhase = '''
            # 20+ lines of bash here
          ''';

          # Use a file
          buildPhase = builtins.readFile ./build.sh;
          ```
          ''
      }
    , tests =
      { valid =
        [ "''short''"
        , ''
          '''
          1
          2
          3
          4
          5
          6
          7
          8
          9
          10'${"'"}''
        , "''single line''"
        ]
      , extra_invalid =
        [ ''
          '''
          1
          2
          3
          4
          5
          6
          7
          8
          9
          10
          11
          12
          13
          14
          15'${"'"}''
        , ''
          checkPhase = '''
            echo 1
            echo 2
            echo 3
            echo 4
            echo 5
            echo 6
            echo 7
            echo 8
            echo 9
            echo 10
            echo 11
            echo 12
          '${"'"}''
        , ''
          installPhase = '''
            echo 1
            echo 2
            echo 3
            echo 4
            echo 5
            echo 6
            echo 7
            echo 8
            echo 9
            echo 10
            echo 11
            echo 12
          '${"'"}''
        ]
      }
    }
