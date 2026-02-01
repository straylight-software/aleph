{ id = "prefer-write-shell-application"
, language = "nix"
, severity = "warning"
, rule =
  { kind = "apply_expression"
  , has =
    { field = "function"
    , kind = "select_expression"
    , has =
      { field = "attrpath"
      , kind = "attrpath"
      , has = { kind = "identifier", regex = "^writeShellScript(Bin)?$" }
      }
    }
  }
, message = "ALEPH-W002: Use writeShellApplication instead of writeShellScript"
, note = Some
    ''
    ## What's wrong?
    `writeShellScript` or `writeShellScriptBin` was used.

    This is discouraged because it:
    - Doesn't provide shell checking
    - Doesn't allow runtime inputs

    ## What can I do to fix this?
    Use `writeShellApplication` instead which provides:
    - Shell syntax checking
    - Runtime dependency management
    - Better error messages
    ''
}
