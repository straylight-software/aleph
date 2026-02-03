let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "no-raw-mkderivation"
    , language = "nix"
    , severity = Severity.Error
    , rule = Schema.Rule::{
      , kind = "select_expression"
      , regex = Some "mkDerivation\$"
      }
    , message = "ALEPH-E010: raw mkDerivation call"
    , note =
        ''
        ## What's wrong?
        Direct `mkDerivation` calls bypass the typed prelude boundary.

        ## What can I do to fix this?
        Use `prelude.mk-derivation` instead.
        ''
    , tests =
      { valid =
        [ "prelude.mk-derivation { name = \"foo\"; }"
        , "aleph.mk-derivation { pname = \"bar\"; }"
        , "myProject.mk-derivation {}"
        ]
      , invalid =
        [ "pkgs.stdenv.mkDerivation { name = \"foo\"; }"
        , "nixpkgs.stdenv.mkDerivation {}"
        , "final.stdenv.mkDerivation { pname = \"baz\"; }"
        ]
      }
    }
