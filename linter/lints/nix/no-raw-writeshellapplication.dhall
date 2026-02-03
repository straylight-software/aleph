let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "no-raw-writeshellapplication"
    , language = "nix"
    , severity = Severity.Error
    , rule = Schema.Rule::{
      , kind = "select_expression"
      , pattern = Some "\$OBJ.writeShellApplication"
      }
    , message = "ALEPH-E012: raw writeShellApplication call"
    , note =
        ''
        ## What's wrong?
        Direct `writeShellApplication` calls bypass the typed prelude boundary.

        ## What can I do to fix this?
        Use `prelude.write-shell-application` instead.
        ''
    , tests =
      { valid =
        [ "prelude.write-shell-application { name = \"foo\"; text = \"echo hi\"; }"
        , "aleph.write-shell-application { name = \"bar\"; text = \"ls\"; }"
        , "myProject.write-shell-application { name = \"script\"; text = \"pwd\"; }"
        ]
      , invalid =
        [ "pkgs.writeShellApplication { name = \"foo\"; text = \"echo hi\"; }"
        , "nixpkgs.writeShellApplication { name = \"bar\"; text = \"test\"; }"
        , "final.writeShellApplication { name = \"script\"; text = \"ls\"; }"
        ]
      }
    }
