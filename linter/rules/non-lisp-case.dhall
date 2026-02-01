{ id = "non-lisp-case"
, language = "nix"
, severity = "warning"
, rule = { kind = "identifier", regex = "[A-Z]" }
, message = "ALEPH-W004: Use lisp-case for identifiers"
, note = Some
    ''
    ## What's wrong?
    Identifier contains uppercase characters.

    This is discouraged because it:
    - Breaks consistency with nixpkgs conventions
    - Makes it harder to remember the exact name

    ## What can I do to fix this?
    Use lisp-case (kebab-case) for all identifiers:

    ```nix
    my-function = ...;  # Good
    myFunction = ...;  # Bad
    ```
    ''
}
