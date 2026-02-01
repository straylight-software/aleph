{ id = "default-nix-in-packages"
, language = "nix"
, severity = "error"
, rule = { kind = "identifier", regex = "^default\\.nix$" }
, message = "ALEPH-E006: Avoid using default.nix in packages"
, note = Some
    ''
    ## What's wrong?
    A `default.nix` file was referenced in the packages directory.

    This is forbidden because it:
    - Makes it harder to navigate the codebase
    - Can lead to confusion about entry points

    ## What can I do to fix this?
    Use explicit file names that describe the package.
    ''
}
