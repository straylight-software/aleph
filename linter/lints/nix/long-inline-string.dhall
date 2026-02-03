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
        ''
        ## What's wrong?
        A multi-line string exceeds 10 lines.

        ## What can I do to fix this?
        Consider moving the content to a file.
        ''
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
      , invalid =
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
          '${"'"}''
        , ''
          '''
          a
          b
          c
          d
          e
          f
          g
          h
          i
          j
          k
          l
          m
          '${"'"}''
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
          10
          11
          12
          13
          14
          15
          '${"'"}''
        ]
      }
    }
