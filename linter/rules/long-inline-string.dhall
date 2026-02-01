{ id = "long-inline-string"
, language = "nix"
, severity = "warning"
, rule = { kind = "string_fragment", regex = "^.{200,}$" }
, message = "ALEPH-W010: Long inline string detected"
, note = Some
    ''
    ## What's wrong?
    An inline string with more than 200 characters was detected.

    This is discouraged because it:
    - Makes code harder to read
    - Is harder to maintain

    ## What can I do to fix this?
    Consider extracting long strings to separate files.
    ''
}
