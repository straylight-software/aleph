let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity
let nodeMatcher = Schema.nodeMatcher

in  { id = "or-null-fallback"
    , language = "nix"
    , severity = Severity.Warning
    , rule =
        { kind = "binary_expression"
        , regex = None Text
        , pattern = Some "$_ or null"
        , has = Some
            ( nodeMatcher
                { kind = None Text
                , field = Some "operator"
                , regex = Some "^\\|$"
                , has = None Schema.NodeMatcher
                , inside = None Schema.NodeMatcher
                }
            )
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-W003: Consider using `? null` default instead of `or null`"
    , note =
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
    , tests =
        { valid =
            [ "{ foo ? null }: foo"
            ]
        , invalid =
            [ "args.foo or null"
            ]
        }
    }
