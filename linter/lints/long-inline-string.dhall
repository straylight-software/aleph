let Schema = ../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "long-inline-string"
    , language = "nix"
    , severity = Severity.Warning
    , rule =
        { kind = "string_fragment"
        , regex = Some "^.{200,}$"
        , pattern = None Text
        , has = None Schema.NodeMatcher
        , not = None { has : Optional Schema.NodeMatcher, inside : Optional Schema.NodeMatcher }
        }
    , message = "ALEPH-W010: Long inline string detected"
    , note =
        ''
        ## What's wrong?
        An inline string with more than 200 characters was detected.

        This is discouraged because it:
        - Makes code harder to read
        - Is harder to maintain

        ## What can I do to fix this?
        Consider extracting long strings to separate files.
        ''
    , tests =
        { valid = [ "short string" ]
        , invalid = 
            [ "this is a very long string that exceeds the limit of two hundred characters and should definitely trigger the linter rule for long inline strings in the codebase because it is way too long"
            ]
        }
    }
