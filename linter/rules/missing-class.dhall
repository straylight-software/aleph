{ id = "missing-class"
, language = "nix"
, severity = "info"
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
    , has = { kind = "attrpath", regex = "^pname$" }
    }
  }
, message = "ALEPH-I001: Missing pname (consider using mkDerivation with pname)"
, note = Some
    ''
    ## What's wrong?
    `mkDerivation` is missing a `pname` attribute.

    This is informational because:
    - Using `pname` with `version` is the modern style
    - It allows automatic inference of the name attribute

    ## What can I do to fix this?
    Consider using `pname` and `version` instead of `name`:

    ```nix
    stdenv.mkDerivation (finalAttrs: {
      pname = "my-package";
      version = "1.0.0";
    })
    ```
    ''
}
