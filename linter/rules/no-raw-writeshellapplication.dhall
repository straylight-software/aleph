{ id = "no-raw-writeshellapplication"
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
      , has = { kind = "identifier", regex = "^writeShellApplication$" }
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
, message = "ALEPH-W005: Use aleph.writeShellApplication"
, note = Some
    ''
    ## What's wrong?
    Raw `writeShellApplication` was used instead of `aleph.writeShellApplication`.

    This is discouraged because it:
    - Doesn't integrate with aleph's conventions
    - May miss aleph-specific optimizations

    ## What can I do to fix this?
    Use `aleph.writeShellApplication` instead.
    ''
}
