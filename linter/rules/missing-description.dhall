{ id = "missing-description"
, language = "nix"
, severity = "warning"
, rule =
  { kind = "apply_expression"
  , has =
    { field = "function"
    , kind = "variable_expression"
    , has = { kind = "identifier", regex = "^mkDerivation$" }
    }
  }
, message = "ALEPH-W009: Missing meta.description"
, note = Some
    ''
    ## What's wrong?
    `mkDerivation` is missing a `meta.description` attribute.

    This is discouraged because it:
    - Makes it harder to understand what the package does
    - Is required for many nixpkgs contributions

    ## What can I do to fix this?
    Add a `meta.description` attribute:

    ```nix
    meta = {
      description = "Brief description of what this package does";
    };
    ```
    ''
}
