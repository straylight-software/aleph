let Schema = ../schemas/Lint.dhall
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
            , "''\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10''"
            , "''single line''"
            ]
        , invalid = 
            [ "''\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n''"
            , "''\na\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\n''"
            , "''\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n''"
            ]
        }
    }
