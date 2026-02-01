{ id = "no-raw-mkderivation"
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
      , has = { kind = "identifier", regex = "^mkDerivation$" }
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
, message = "ALEPH-W007: Use aleph.mkDerivation"
, note = Some
    ''
    ## What's wrong?
    Raw `mkDerivation` was used instead of `aleph.mkDerivation`.

    This is discouraged because it:
    - Doesn't integrate with aleph's conventions
    - May miss aleph-specific features

    ## What can I do to fix this?
    Use `aleph.mkDerivation` instead.
    ''
}
