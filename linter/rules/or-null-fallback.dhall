{ id = "or-null-fallback"
, language = "nix"
, severity = "warning"
, rule =
  { kind = "binary_expression"
  , has = { field = "operator", regex = "^\\|$" }
  , pattern = "$_ or null"
  }
, message = "ALEPH-W003: Consider using `? null` default instead of `or null`"
, note = Some
    ''
    ## What's wrong?
    Using `or null` as a fallback pattern.

    This is discouraged because it:
    - Is less idiomatic than using default values
    - Can mask missing values

    ## What can I do to fix this?
    Consider using `? null` in the function arguments instead:

    ```nix
    { foo ? null }: ...
    ```
    ''
}
