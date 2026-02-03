let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "prefer-write-shell-application"
    , language = "nix"
    , severity = Severity.Warning
    , rule =
        { kind = "apply_expression"
        , regex = None Text
        , pattern = None Text
        , has = Some
            ( nodeMatcher
                { kind = Some "identifier"
                , field = Some "function"
                , regex = Some "^writeShellScript(Bin)?$"
                , has = None Schema.NodeMatcher
                , inside = None Schema.NodeMatcher
                }
            )
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-W002: Use writeShellApplication instead of writeShellScript"
    , note =
        ''
        ## What's wrong?
        `writeShellScript` or `writeShellScriptBin` was used.

        This is discouraged because it:
        - Doesn't provide shell checking
        - Doesn't allow runtime inputs

        ## What can I do to fix this?
        Use `writeShellApplication` instead which provides:
        - Shell syntax checking
        - Runtime dependency management
        - Better error messages
        ''
    , tests =
        { valid = [ "aleph.writeShellApplication { name = \"foo\"; text = \"echo hi\"; }" ]
        , invalid = 
            [ "writeShellScript \"foo\" \"echo hi\""
            , "writeShellScriptBin \"foo\" \"echo hi\""
            ]
        }
    }
