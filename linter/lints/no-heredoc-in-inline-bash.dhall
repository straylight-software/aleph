let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "no-heredoc-in-inline-bash"
    , language = "nix"
    , severity = Severity.Error
    , rule =
        { kind = "string_fragment"
        , regex = Some "<<.*<<"
        , pattern = None Text
        , has = None Schema.NodeMatcher
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-E005: Avoid heredocs in inline bash"
    , note =
        ''
        ## What's wrong?
        Heredoc syntax detected in inline bash string.

        This is forbidden because it:
        - Is fragile and hard to read
        - Can cause quoting issues

        ## What can I do to fix this?
        Extract the script to a separate file using `writeShellApplication`.
        ''
    , tests =
        { valid =
            [ "echo hello"
            ]
        , invalid =
            [ ''
              cat <<EOF
              hello
              EOF
              ''
            ]
        }
    }
