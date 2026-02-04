let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

let node = Schema.someNodeMatcher

let Match = Schema.NodeMatcherWithDefaults

in  { id = "prefer-write-shell-application"
    , language = "nix"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "apply_expression"
      , any = Some
        [ Schema.SubRule::{
          , field = Some "function"
          , kind = Some "select_expression"
          , has =
              node
                Match::{
                , kind = Some "attrpath"
                , field = Some "attrpath"
                , regex = Some "^(writeShellScript|writeShellScriptBin)\$"
                }
          }
        , Schema.SubRule::{
          , field = Some "function"
          , kind = Some "variable_expression"
          , has =
              node
                Match::{
                , kind = Some "identifier"
                , regex = Some "^(writeShellScript|writeShellScriptBin)\$"
                }
          }
        ]
      }
    , message = "ALEPH-W006: prefer writeShellApplication"
    , note =
        { description = "`writeShellScript` and `writeShellScriptBin` are deprecated."
        , examples =
          [ "writeShellScript \"foo\" \"echo hi\""
          , "writeShellScriptBin \"foo\" \"echo hi\""
          , "pkgs.writeShellScript \"bar\" \"ls\""
          ]
        , suggested_fix =
            ''
            Use `writeShellApplication` instead.
            ''
        }
    , tests =
      { valid =
        [ "writeShellApplication { name = \"foo\"; text = \"echo hi\"; }"
        , "writeShellApplication { name = \"script\"; text = \"ls -la\"; }"
        , "writeShellApplication { name = \"checker\"; text = \"test -f file\"; }"
        ]
      , extra_invalid = [] : List Text
      }
    }
