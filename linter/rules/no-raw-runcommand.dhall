{ id = "no-raw-runcommand"
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
      , has = { kind = "identifier", regex = "^runCommand$" }
      }
    }
  , not.inside =
    { kind = "select_expression"
    , has =
      { field = "attrpath"
      , kind = "attrpath"
      , has = { kind = "identifier", regex = "^aleph$" }
      }
    }
  }
, message = "ALEPH-W006: Use aleph.runCommand"
, note = Some
    ''
    ## What's wrong?
    Raw `runCommand` was used instead of `aleph.runCommand`.

    This is discouraged because it:
    - Doesn't integrate with aleph's conventions
    - May miss aleph-specific features

    ## What can I do to fix this?
    Use `aleph.runCommand` instead.
    ''
}
