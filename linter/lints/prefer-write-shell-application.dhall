let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "prefer-write-shell-application"
    , language = "nix"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "apply_expression"
      , any = Some
          [ Schema.SubRule::{
            , field = Some "function"
            , kind = Some "select_expression"
            , has = Some
                ( nodeMatcher
                    { kind = Some "attrpath"
                    , field = Some "attrpath"
                    , regex = Some "^(writeShellScript|writeShellScriptBin)$"
                    , has = None Schema.NodeMatcher
                    , inside = None Schema.NodeMatcher
                    }
                )
            }
          , Schema.SubRule::{
            , field = Some "function"
            , kind = Some "variable_expression"
            , has = Some
                ( nodeMatcher
                    { kind = Some "identifier"
                    , field = None Text
                    , regex = Some "^(writeShellScript|writeShellScriptBin)$"
                    , has = None Schema.NodeMatcher
                    , inside = None Schema.NodeMatcher
                    }
                )
            }
          ]
      }
    , message = "ALEPH-W006: prefer writeShellApplication"
    , note =
        ''
        ## What's wrong?
        `writeShellScript` and `writeShellScriptBin` are deprecated.

        ## What can I do to fix this?
        Use `writeShellApplication` instead.
        ''
    , tests =
        { valid = [ "writeShellApplication { name = \"foo\"; text = \"echo hi\"; }" ]
        , invalid = 
            [ "writeShellScript \"foo\" \"echo hi\""
            , "writeShellScriptBin \"foo\" \"echo hi\""
            ]
        }
    }
