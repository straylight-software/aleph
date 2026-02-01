{ id = "no-heredoc-in-inline-bash"
, language = "nix"
, severity = "error"
, rule = { kind = "string_fragment", regex = "<<.*<<" }
, message = "ALEPH-E005: Avoid heredocs in inline bash"
, note = Some
    ''
    ## What's wrong?
    Heredoc syntax detected in inline bash string.

    This is forbidden because it:
    - Is fragile and hard to read
    - Can cause quoting issues

    ## What can I do to fix this?
    Extract the script to a separate file using `writeShellApplication`.
    ''
}
