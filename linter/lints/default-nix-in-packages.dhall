let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "default-nix-in-packages"
    , language = "nix"
    , severity = Severity.Error
    , rule =
        { kind = "identifier"
        , regex = Some "^default\\.nix$"
        , pattern = None Text
        , has = None Schema.NodeMatcher
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-E006: Avoid using default.nix in packages"
    , note =
        ''
        ## What's wrong?
        A `default.nix` file was referenced in the packages directory.

        This is forbidden because it:
        - Makes it harder to navigate the codebase
        - Can lead to confusion about entry points

        ## What can I do to fix this?
        Use explicit file names that describe the package.
        ''
    , tests =
        { valid =
            [ "hello = callPackage ./hello.nix { };"
            ]
        , invalid =
            [ "hello = callPackage ./default.nix { };"
            ]
        }
    }
