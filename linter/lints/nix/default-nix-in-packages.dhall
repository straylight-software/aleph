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
        { description = "A `default.nix` file was referenced in the packages directory."
        , examples =
          [ "\"./default.nix\""
          , "\"nix/packages/bar/default.nix\""
          , "\"./package/default.nix\""
          ]
        , suggested_fix =
            ''
            Use explicit file names that describe the package.
            ''
        }
    , tests =
      { valid =
        [ "\"./hello.nix\""
        , "\"./my-package.nix\""
        , "\"nix/packages/foo/package.nix\""
        ]
      , extra_invalid = [] : List Text
      }
    }
