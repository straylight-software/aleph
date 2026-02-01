{ id = "missing-meta"
, language = "nix"
, severity = "warning"
, rule =
  { kind = "apply_expression"
  , has =
    { field = "function"
    , kind = "variable_expression"
    , has = { kind = "identifier", regex = "^mkDerivation$" }
    }
  , not.has =
    { field = "argument"
    , kind = "attrset_expression"
    , has = { kind = "attrpath", regex = "^meta$" }
    }
  }
, message = "ALEPH-W008: Missing meta attribute"
, note = Some
    ''
    ## What's wrong?
    `mkDerivation` is missing a `meta` attribute.

    This is discouraged because it:
    - Omits important package metadata
    - Makes it harder to find information about the package

    ## What can I do to fix this?
    Add a `meta` attribute with at least basic information:

    ```nix
    meta = {
      description = "Brief description";
      license = lib.licenses.mit;
    };
    ```
    ''
}
