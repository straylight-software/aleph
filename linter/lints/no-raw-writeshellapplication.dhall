let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "no-raw-writeshellapplication"
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
                , regex = Some "^writeShellApplication$"
                , has = None Schema.NodeMatcher
                , inside = None Schema.NodeMatcher
                }
            )
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-W005: Use aleph.writeShellApplication"
    , note =
        ''
        ## What's wrong?
        Raw `writeShellApplication` was used instead of `aleph.writeShellApplication`.

        This is discouraged because it:
        - Doesn't integrate with aleph's conventions
        - May miss aleph-specific optimizations

        ## What can I do to fix this?
        Use `aleph.writeShellApplication` instead.
        ''
    , tests =
        { valid = [ "aleph.writeShellApplication { name = \"foo\"; text = \"echo hi\"; }" ]
        , invalid = [ "writeShellApplication { name = \"foo\"; text = \"echo hi\"; }" ]
        }
    }
