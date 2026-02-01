{ id = "rec-anywhere"
, language = "nix"
, severity = "warning"
, rule.kind = "rec_attrset_expression"
, message = "ALEPH-W001: `rec` usage detected"
, note = Some
    ''
    ## What's wrong?
    A recursive attrset (`rec { ... }`) was detected.

    This is discouraged because it:
    - Makes it harder to reason about evaluation order
    - Can lead to infinite recursion bugs

    ## What can I do to fix this?
    Consider using `let ... in` bindings or the fixed-point pattern instead.
    ''
}
