let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "default-nix-in-packages"
    , language = "nix"
    , severity = Severity.Error
    , rule = Schema.Rule::{
      , kind = "string_fragment"
      , regex = Some "default\\.nix"
      }
    , message = "ALEPH-E006: Avoid using default.nix in packages"
    , note =
        ''
        ## What's wrong?
        A `default.nix` file was referenced in the packages directory.

        ## What can I do to fix this?
        Use explicit file names that describe the package.
        ''
    , tests =
      { valid =
        [ "\"./hello.nix\""
        , "\"./my-package.nix\""
        , "\"nix/packages/foo/package.nix\""
        ]
      , invalid =
        [ "\"./default.nix\""
        , "\"nix/packages/bar/default.nix\""
        , "\"./package/default.nix\""
        ]
      }
    }
