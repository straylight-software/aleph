{- Detects `with lib;` usage in nix files.

This rule matches "with lib;" expressions and reports them as errors
since they obscure name origins and break IDE tooling.
-}
let Schema = ../schemas/Lint.dhall
let nodeMatcher = Schema.nodeMatcher

in  { id = "with-lib"
    , language = "nix"
    , severity = Schema.Severity.Error
    , rule = Schema.Rule::{
      , kind = "with_expression"
      , has = Some
          ( nodeMatcher
              { kind = Some "variable_expression"
              , field = Some "environment"
              , regex = None Text
              , has = Some
                  ( nodeMatcher
                      { kind = Some "identifier"
                      , field = None Text
                      , regex = Some "^lib$"
                      , has = None Schema.NodeMatcher
                      , inside = None Schema.NodeMatcher
                      }
                  )
              , inside = None Schema.NodeMatcher
              }
          )
      }
    , message = "ALEPH-E001: `with lib;` statement"
    , note =
        ''
        ## What's wrong?
        Usage of the `with lib;` construct was detected.

        ## What can I do to fix this?
        Use explicit imports instead:

        ```nix
        inherit (lib) types mkOption;
        ```
        ''
    , tests.valid = [ "environment.systemPackages = with pkgs; [ vim git ];" ]
    , tests.invalid = [ "with lib;\n{ options.foo = mkOption { }; }" ]
    } : Schema.Lint
