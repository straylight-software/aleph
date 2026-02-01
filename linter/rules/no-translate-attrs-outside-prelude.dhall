{ id = "no-translate-attrs-outside-prelude"
, language = "nix"
, severity = "error"
, rule =
  { kind = "select_expression"
  , has =
    { field = "attrpath"
    , kind = "attrpath"
    , has = { kind = "identifier", regex = "^translateAttrs$" }
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
, message = "ALEPH-E003: translateAttrs should only be used in prelude"
, note = Some
    ''
    ## What's wrong?
    `translateAttrs` was used outside of the prelude.

    This is forbidden because it:
    - Is an internal prelude function
    - Should not be used in user code

    ## What can I do to fix this?
    Use standard attribute manipulation functions instead.
    ''
}
