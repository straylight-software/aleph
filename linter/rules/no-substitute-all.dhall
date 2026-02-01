{ id = "no-substitute-all"
, language = "nix"
, severity = "error"
, rule =
  { kind = "apply_expression"
  , has =
    { field = "function"
    , kind = "select_expression"
    , has =
      { field = "attrpath"
      , kind = "attrpath"
      , has = { kind = "identifier", regex = "^substituteAll$" }
      }
    }
  }
, message = "ALEPH-E004: Avoid using substituteAll"
, note = Some
    ''
    ## What's wrong?
    `substituteAll` was used.

    This is forbidden because it:
    - Is fragile and hard to debug
    - Makes code harder to understand

    ## What can I do to fix this?
    Use explicit string interpolation or structured arguments instead.
    ''
}
