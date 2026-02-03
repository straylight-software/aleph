let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "no-translate-attrs-outside-prelude"
    , language = "nix"
    , severity = Severity.Error
    , rule =
        { kind = "select_expression"
        , regex = None Text
        , pattern = None Text
        , has = Some
            ( nodeMatcher
                { kind = Some "identifier"
                , field = Some "attrpath"
                , regex = Some "^translateAttrs$"
                , has = None Schema.NodeMatcher
                , inside = None Schema.NodeMatcher
                }
            )
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-E003: translateAttrs should only be used in prelude"
    , note =
        ''
        ## What's wrong?
        `translateAttrs` was used outside of the prelude.

        This is forbidden because it:
        - Is an internal prelude function
        - Should not be used in user code

        ## What can I do to fix this?
        Use standard attribute manipulation functions instead.
        ''
    , tests =
        { valid = [ "lib.translateAttrs {}" ]
        , invalid = [ "translateAttrs {}" ]
        }
    }
