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
        { description = "Direct `mkDerivation` calls bypass the typed prelude boundary."
        , examples =
          [ "pkgs.stdenv.mkDerivation { name = \"foo\"; }"
          , "nixpkgs.stdenv.mkDerivation {}"
          , "final.stdenv.mkDerivation { pname = \"baz\"; }"
          ]
        , suggested_fix =
            ''
            Use `prelude.mk-derivation` instead.
            ''
        }
    , tests =
      { valid =
        [ "prelude.mk-derivation { name = \"foo\"; }"
        , "aleph.mk-derivation { pname = \"bar\"; }"
        , "myProject.mk-derivation {}"
        ]
      , extra_invalid = [] : List Text
      }
    }
