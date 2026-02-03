let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "no-raw-runcommand"
    , language = "nix"
    , severity = Severity.Warning
    , rule =
        { kind = "apply_expression"
        , regex = None Text
        , pattern = None Text
        , has = Some
            ( nodeMatcher
                { kind = Some "identifier"
                , field = Some "function"
                , regex = Some "^runCommand$"
                , has = None Schema.NodeMatcher
                , inside = None Schema.NodeMatcher
                }
            )
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-W006: Use aleph.runCommand"
    , note =
        ''
        ## What's wrong?
        Raw `runCommand` was used instead of `aleph.runCommand`.

        This is discouraged because it:
        - Doesn't integrate with aleph's conventions
        - May miss aleph-specific features

        ## What can I do to fix this?
        Use `aleph.runCommand` instead.
        ''
    , tests =
        { valid = [ "aleph.runCommand \"foo\" {} \"echo hi\"" ]
        , invalid = [ "runCommand \"foo\" {} \"echo hi\"" ]
        }
    }
