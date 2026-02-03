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

let Severity = < Error | Warning | Info | Hint >

let severityToText =
      λ(s : Severity) →
        merge
          { Error = "error", Warning = "warning", Info = "info", Hint = "hint" }
          s

let NodeMatcherF =
      λ(r : Type) →
        { kind : Optional Text
        , field : Optional Text
        , regex : Optional Text
        , has : Optional r
        , inside : Optional r
        }

let NodeMatcher = ∀(r : Type) → (NodeMatcherF r → r) → r

let foldNodeMatcher =
      λ(r : Type) → λ(f : NodeMatcherF r → r) → λ(m : NodeMatcher) → m r f

let nodeMatcher
    : NodeMatcherF NodeMatcher → NodeMatcher
    = λ(fm : NodeMatcherF NodeMatcher) →
      λ(r : Type) →
      λ(f : NodeMatcherF r → r) →
        let mapOpt = Prelude.Optional.map

        in  f
              { kind = fm.kind
              , field = fm.field
              , regex = fm.regex
              , has = mapOpt NodeMatcher r (foldNodeMatcher r f) fm.has
              , inside = mapOpt NodeMatcher r (foldNodeMatcher r f) fm.inside
              }

let NodeMatcherWithDefaults =
      { Type = NodeMatcherF NodeMatcher
      , default =
        { kind = None Text
        , field = None Text
        , regex = None Text
        , has = None NodeMatcher
        , inside = None NodeMatcher
        }
      }

let someNodeMatcher
    : NodeMatcherF NodeMatcher → Optional NodeMatcher
    = λ(fm : NodeMatcherF NodeMatcher) → Some (nodeMatcher fm)

let SubRule =
      { Type =
          { kind : Optional Text
          , field : Optional Text
          , regex : Optional Text
          , has : Optional NodeMatcher
          }
      , default =
        { kind = None Text
        , field = None Text
        , regex = None Text
        , has = None NodeMatcher
        }
      }

let RuleNot =
      { has : Optional NodeMatcher
      , inside : Optional NodeMatcher
      , regex : Optional Text
      }

let Rule =
      { Type =
          { kind : Text
          , regex : Optional Text
          , pattern : Optional Text
          , has : Optional NodeMatcher
          , not : Optional RuleNot
          , all : Optional (List SubRule.Type)
          , any : Optional (List SubRule.Type)
          }
      , default =
        { kind = ""
        , regex = None Text
        , pattern = None Text
        , has = None NodeMatcher
        , not = None RuleNot
        , all = None (List SubRule.Type)
        , any = None (List SubRule.Type)
        }
      }

let Lint =
      { id : Text
      , language : Text
      , severity : Severity
      , rule : Rule.Type
      , message : Text
      , note : Text
      , tests : { valid : List Text, invalid : List Text }
      }

let toList = Prelude.Optional.toList

let JSONField = { mapKey : Text, mapValue : JSON.Type }

let maybeField =
      λ(name : Text) →
      λ(render : Text → JSON.Type) →
      λ(value : Optional Text) →
        let makeField = λ(v : Text) → { mapKey = name, mapValue = render v }

        in  toList
              JSONField
              (Prelude.Optional.map Text JSONField makeField value)

let maybeNodeField =
      λ(name : Text) →
      λ(value : Optional JSON.Type) →
        let makeField = λ(v : JSON.Type) → { mapKey = name, mapValue = v }

        in  toList
              JSONField
              (Prelude.Optional.map JSON.Type JSONField makeField value)

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

let nodeMatcherToJSON
    : NodeMatcher → JSON.Type
    = let foldFn =
            λ(fm : NodeMatcherF JSON.Type) → JSON.object (nodeMatcherFToJSON fm)

      in  foldNodeMatcher JSON.Type foldFn

let maybeNodeMatcherField =
      λ(name : Text) →
      λ(value : Optional NodeMatcher) →
        let makeField =
              λ(v : NodeMatcher) →
                { mapKey = name, mapValue = nodeMatcherToJSON v }

        in  toList
              JSONField
              (Prelude.Optional.map NodeMatcher JSONField makeField value)

let notFieldsToJSON =
      λ(n : RuleNot) →
        Prelude.List.concat
          JSONField
          [ maybeNodeMatcherField "has" n.has
          , maybeNodeMatcherField "inside" n.inside
          , maybeField "regex" JSON.string n.regex
          ]

let subRuleToJSON
    : SubRule.Type → JSON.Type
    = λ(sr : SubRule.Type) →
        let innerFields =
              Prelude.List.concat
                JSONField
                [ maybeField "kind" JSON.string sr.kind
                , maybeField "field" JSON.string sr.field
                , maybeField "regex" JSON.string sr.regex
                , maybeNodeMatcherField "has" sr.has
                ]

        in  JSON.object
              [ { mapKey = "has", mapValue = JSON.object innerFields } ]

let subRulesListToJSON
    : List SubRule.Type → JSON.Type
    = λ(rules : List SubRule.Type) →
        JSON.array (Prelude.List.map SubRule.Type JSON.Type subRuleToJSON rules)

let ruleToJSON
    : Rule.Type → JSON.Type
    = λ(rule : Rule.Type) →
        let makeNotField =
              λ(n : RuleNot) →
                { mapKey = "not", mapValue = JSON.object (notFieldsToJSON n) }

        in  JSON.object
              ( Prelude.List.concat
                  JSONField
                  [ [ { mapKey = "kind", mapValue = JSON.string rule.kind } ]
                  , maybeField "regex" JSON.string rule.regex
                  , maybeField "pattern" JSON.string rule.pattern
                  , maybeNodeMatcherField "has" rule.has
                  , toList
                      JSONField
                      ( Prelude.Optional.map
                          RuleNot
                          JSONField
                          makeNotField
                          rule.not
                      )
                  , toList
                      JSONField
                      ( Prelude.Optional.map
                          (List SubRule.Type)
                          JSONField
                          ( λ(rules : List SubRule.Type) →
                              { mapKey = "all"
                              , mapValue = subRulesListToJSON rules
                              }
                          )
                          rule.all
                      )
                  , toList
                      JSONField
                      ( Prelude.Optional.map
                          (List SubRule.Type)
                          JSONField
                          ( λ(rules : List SubRule.Type) →
                              { mapKey = "any"
                              , mapValue = subRulesListToJSON rules
                              }
                          )
                          rule.any
                      )
                  ]
              )

let renderRuleYAML
    : Lint → Text
    = λ(lint : Lint) →
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

let renderTestYAML
    : Lint → Text
    = λ(lint : Lint) →
        JSON.renderYAML
          ( JSON.object
              ( toMap
                  { id = JSON.string lint.id
                  , valid =
                      JSON.array
                        ( Prelude.List.map
                            Text
                            JSON.Type
                            JSON.string
                            lint.tests.valid
                        )
                  , invalid =
                      JSON.array
                        ( Prelude.List.map
                            Text
                            JSON.Type
                            JSON.string
                            lint.tests.invalid
                        )
                  }
              )
          )

let renderSGConfigYAML
    : Text
    = JSON.renderYAML
        ( JSON.object
            ( toMap
                { ruleDirs = JSON.array [ JSON.string "./rules" ]
                , testConfigs =
                    JSON.array
                      [ JSON.object
                          (toMap { testDir = JSON.string "./rule-tests" })
                      ]
                }
            )
        )

in  { Severity
    , severityToText
    , NodeMatcherF
    , NodeMatcher
    , nodeMatcher
    , someNodeMatcher
    , foldNodeMatcher
    , NodeMatcherWithDefaults
    , SubRule
    , RuleNot
    , Rule
    , Lint
    , renderRuleYAML
    , renderTestYAML
    , renderSGConfigYAML
    }
