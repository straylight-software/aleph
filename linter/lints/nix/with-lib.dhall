let Schema = ../../schemas/Lint.dhall

let node = Schema.someNodeMatcher

let Match = Schema.NodeMatcherWithDefaults

in    { id = "with-lib"
      , language = "nix"
      , severity = Schema.Severity.Error
      , rule = Schema.Rule::{
        , kind = "with_expression"
        , has =
            node
              Match::{
              , kind = Some "variable_expression"
              , field = Some "environment"
              , has =
                  node
                    Match::{ kind = Some "identifier", regex = Some "^lib\$" }
              }
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
      , tests =
        { valid =
          [ "environment.systemPackages = with pkgs; [ vim git ];"
          , "with pkgs; [ hello git ]"
          , "with builtins; map toString [ 1 2 3 ]"
          ]
        , invalid =
          [ ''
            with lib;
            { options.foo = mkOption { }; }''
          , "with lib; [ types str ]"
          , "with lib; mkOption { type = types.str; }"
          ]
        }
      }
    : Schema.Lint
