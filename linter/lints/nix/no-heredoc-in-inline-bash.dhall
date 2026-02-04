let Schema = ../../schemas/Lint.dhall

let Severity = Schema.Severity

in  { id = "no-heredoc-in-inline-bash"
    , language = "nix"
    , severity = Severity.Error
    , rule = Schema.Rule::{ kind = "string_fragment", regex = Some "<<" }
    , message = "ALEPH-E006: heredoc in inline bash string"
    , note =
        { description = "Inline bash strings that use heredocs are fragile and hard to lint safely."
        , examples =
          [ "''cat <<EOF''"
          , "''tee <<INPUT''"
          , "''bash <<SCRIPT''"
          ]
        , suggested_fix =
            ''
            Consider moving the content to a file.
            ''
        }
    , tests =
      { valid = [ "''echo hello''", "''ls -la''", "''pwd && ls''" ]
      , extra_invalid = [] : List Text
      }
    }
