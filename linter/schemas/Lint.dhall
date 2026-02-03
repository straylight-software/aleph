{-
Schema for ast-grep lint rules using Church-encoded recursive types.

This module provides types and functions for defining ast-grep lint rules
with integrated test cases. The recursive NodeMatcher type is encoded using
Church encoding since Dhall doesn't support direct recursion.

Example usage:

```dhall
let Schema = ../schemas/Lint.dhall

in  { id = "my-rule"
    , rule = Schema.Rule::{ kind = "identifier", has = Some matcher }
    , ...
    }
```

References:
- https://docs.dhall-lang.org/howtos/How-to-translate-recursive-code-to-Dhall.html
- https://ast-grep.github.io/guide/rule-config.html
-}

let Prelude =
      https://prelude.dhall-lang.org/v20.1.0/package.dhall
        sha256:26b0ef498663d269e4dc6a82b0ee289ec565d683ef4c00d0ebdd25333a5a3c98

let JSON = Prelude.JSON

-- | Severity levels matching ast-grep's severity enum
let Severity = < Error | Warning | Info | Hint >

-- | Convert severity to ast-grep's text representation
let severityToText =
      λ(s : Severity) →
        merge
          { Error = "error"
          , Warning = "warning"
          , Info = "info"
          , Hint = "hint"
          }
          s

-- | Recursion scheme for NodeMatcher (Step 1 of Church encoding)
let NodeMatcherF = λ(r : Type) →
      { kind : Optional Text
      , field : Optional Text
      , regex : Optional Text
      , has : Optional r
      , inside : Optional r
      }

-- | Church-encoded recursive type for AST node matchers
-- |
-- | A value of this type is a higher-order function that takes:
-- | - A result type `r`
-- | - An algebra `NodeMatcherF r → r`
-- | and produces a value of type `r`
let NodeMatcher = ∀(r : Type) → (NodeMatcherF r → r) → r

-- | Fold function for consuming NodeMatcher values
let foldNodeMatcher =
      λ(r : Type) → λ(f : NodeMatcherF r → r) → λ(m : NodeMatcher) → m r f

-- | Smart constructor for building NodeMatcher values
let nodeMatcher : NodeMatcherF NodeMatcher → NodeMatcher =
      λ(fm : NodeMatcherF NodeMatcher) →
        λ(r : Type) → λ(f : NodeMatcherF r → r) →
          let mapOpt = Prelude.Optional.map
          in  f
                { kind = fm.kind
                , field = fm.field
                , regex = fm.regex
                , has = mapOpt NodeMatcher r (foldNodeMatcher r f) fm.has
                , inside = mapOpt NodeMatcher r (foldNodeMatcher r f) fm.inside
                }

-- | Rule configuration type with default values
let Rule =
      { Type =
          { kind : Text
          , regex : Optional Text
          , pattern : Optional Text
          , has : Optional NodeMatcher
          , not : Optional { has : Optional NodeMatcher, inside : Optional NodeMatcher }
          }
      , default =
          { kind = ""
          , regex = None Text
          , pattern = None Text
          , has = None NodeMatcher
          , not = None { has : Optional NodeMatcher, inside : Optional NodeMatcher }
          }
      }

-- | Complete lint rule definition
let Lint =
      { id : Text
      , language : Text
      , severity : Severity
      , rule : Rule.Type
      , message : Text
      , note : Text
      , tests : { valid : List Text, invalid : List Text }
      }

-- | Convert Optional to singleton List
let toList = Prelude.Optional.toList

-- | Type alias for JSON object fields
let JSONField = { mapKey : Text, mapValue : JSON.Type }

-- | Helper to optionally add a JSON field
let maybeField =
      λ(name : Text) →
      λ(render : Text → JSON.Type) →
      λ(value : Optional Text) →
        let makeField = λ(v : Text) → { mapKey = name, mapValue = render v }
        in  toList JSONField (Prelude.Optional.map Text JSONField makeField value)

-- | Helper to optionally add a nested JSON field
let maybeNodeField =
      λ(name : Text) →
      λ(value : Optional JSON.Type) →
        let makeField = λ(v : JSON.Type) → { mapKey = name, mapValue = v }
        in  toList JSONField (Prelude.Optional.map JSON.Type JSONField makeField value)

-- | Convert NodeMatcherF to JSON object fields
let nodeMatcherFToJSON =
      λ(fm : NodeMatcherF JSON.Type) →
        Prelude.List.concat
          JSONField
          [ maybeField "kind" JSON.string fm.kind
          , maybeField "field" JSON.string fm.field
          , maybeField "regex" JSON.string fm.regex
          , maybeNodeField "has" fm.has
          , maybeNodeField "inside" fm.inside
          ]

-- | Recursively convert NodeMatcher to JSON
let nodeMatcherToJSON : NodeMatcher → JSON.Type =
      let foldFn = λ(fm : NodeMatcherF JSON.Type) → JSON.object (nodeMatcherFToJSON fm)
      in  foldNodeMatcher JSON.Type foldFn

-- | Helper to optionally add a nested NodeMatcher field
let maybeNodeMatcherField =
      λ(name : Text) →
      λ(value : Optional NodeMatcher) →
        let makeField = λ(v : NodeMatcher) → { mapKey = name, mapValue = nodeMatcherToJSON v }
        in  toList JSONField (Prelude.Optional.map NodeMatcher JSONField makeField value)

-- | Convert negation fields to JSON
let notFieldsToJSON =
      λ(n : { has : Optional NodeMatcher, inside : Optional NodeMatcher }) →
        Prelude.List.concat JSONField [ maybeNodeMatcherField "has" n.has, maybeNodeMatcherField "inside" n.inside ]

-- | Convert Rule to JSON object
let ruleToJSON : Rule.Type → JSON.Type =
      λ(rule : Rule.Type) →
        let notType = { has : Optional NodeMatcher, inside : Optional NodeMatcher }
        let makeNotField =
              λ(n : notType) → { mapKey = "not", mapValue = JSON.object (notFieldsToJSON n) }
        in  JSON.object
              ( Prelude.List.concat
                  JSONField
                  [ [ { mapKey = "kind", mapValue = JSON.string rule.kind } ]
                  , maybeField "regex" JSON.string rule.regex
                  , maybeField "pattern" JSON.string rule.pattern
                  , maybeNodeMatcherField "has" rule.has
                  , toList JSONField (Prelude.Optional.map notType JSONField makeNotField rule.not)
                  ]
              )

-- | Render a lint rule to YAML format
let renderRuleYAML : Lint → Text =
      λ(lint : Lint) →
        JSON.renderYAML
          ( JSON.object
              ( toMap
                  { id = JSON.string lint.id
                  , language = JSON.string lint.language
                  , severity = JSON.string (severityToText lint.severity)
                  , message = JSON.string lint.message
                  , note = JSON.string lint.note
                  , rule = ruleToJSON lint.rule
                  }
              )
          )

-- | Render test cases to YAML format
let renderTestYAML : Lint → Text =
      λ(lint : Lint) →
        JSON.renderYAML
          ( JSON.object
              ( toMap
                  { id = JSON.string lint.id
                  , valid = JSON.array (Prelude.List.map Text JSON.Type JSON.string lint.tests.valid)
                  , invalid = JSON.array (Prelude.List.map Text JSON.Type JSON.string lint.tests.invalid)
                  }
              )
          )

-- | Render the main sgconfig.yaml content
let renderSGConfigYAML : Text =
      JSON.renderYAML
        ( JSON.object
            ( toMap
                { ruleDirs = JSON.array [ JSON.string "./rules" ]
                , testConfigs = JSON.array [ JSON.object (toMap { testDir = JSON.string "./tests" }) ]
                }
            )
        )

-- | Export all public types and functions
in  { Severity
    , severityToText
    , NodeMatcherF
    , NodeMatcher
    , nodeMatcher
    , foldNodeMatcher
    , Rule
    , Lint
    , renderRuleYAML
    , renderTestYAML
    , renderSGConfigYAML
    }
