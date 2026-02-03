let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "no-raw-runcommand"
    , language = "nix"
    , severity = Severity.Error
    , rule = Schema.Rule::{
      , kind = "select_expression"
      , pattern = Some "\$OBJ.runCommand"
      }
    , message = "ALEPH-E011: raw runCommand call"
    , note =
        ''
        ## What's wrong?
        Direct `runCommand` calls bypass the typed prelude boundary.

        ## What can I do to fix this?
        Use `prelude.run-command` instead.
        ''
    , tests =
      { valid =
        [ "prelude.run-command \"foo\" { } \"echo hi\""
        , "aleph.run-command \"bar\" {} \"ls\""
        , "myProject.run-command \"script\" {} \"pwd\""
        ]
      , invalid =
        [ "pkgs.runCommand \"foo\" { } \"echo hi\""
        , "nixpkgs.runCommand \"bar\" {} \"test\""
        , "final.runCommand \"script\" {} \"ls\""
        ]
      }
    }
