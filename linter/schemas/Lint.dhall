{- 
Ast-grep Lint Schema for Dhall
==============================

This schema provides types and helpers for defining ast-grep lint rules
with integrated test cases in a single Dhall file.

Usage in lint files:
--------------------
    let Schema = ../schemas/Lint.dhall
    let Severity = Schema.Severity
    
    in  { id = "my-rule"
        , language = "nix"
        , severity = Severity.Warning
        , rule = 
            { kind = "identifier"
            , regex = Some "^foo$"
            , has = Some
                (Schema.nodeMatcher
                    { kind = Some "string"
                    , field = None Text
                    , regex = Some "bar"
                    , has = None (Schema.NodeMatcherF Schema.NodeMatcher)
                    }
                )
            }
        , message = "Found foo"
        , note = "This is bad because..."
        , tests = { valid = ["bar"], invalid = ["foo"] }
        }

Required Fields:
----------------
- id: Text                          Unique identifier for the rule
- language: Text                    Target language (e.g., "nix")
- severity: Severity                One of: Error, Warning, Info, Hint
- rule: Rule                        The ast-grep rule definition (record)
- message: Text                     Short message shown when rule matches  
- note: Text                        Detailed documentation (markdown supported)
- tests:                            Test cases for the rule
  - valid: List Text                At least one valid (non-matching) example
  - invalid: List Text              At least one invalid (matching) example

Output:
-------
The generator produces separate YAML files for ast-grep:
- sgconfig.yaml  - Ast-grep configuration
- rules/*.yml    - Rule definitions
- tests/*.yml    - Test cases

References:
-----------
- Ast-grep severity docs: https://ast-grep.github.io/guide/project/severity.html
- Severity values: error, warning, info, hint (and 'off' to disable)
- Dhall Church encoding: https://docs.dhall-lang.org/howtos/How-to-translate-recursive-code-to-Dhall.html
-}

let Prelude =
      https://prelude.dhall-lang.org/v20.1.0/package.dhall
        sha256:26b0ef498663d269e4dc6a82b0ee289ec565d683ef4c00d0ebdd25333a5a3c98

let JSON = Prelude.JSON

let Map = Prelude.Map.Type



{- Severity levels as defined by ast-grep. -}
let Severity =
      < Error | Warning | Info | Hint >

{- Convert Severity enum to Text for YAML output. -}
let severityToText =
      λ(s : Severity) →
        merge
          { Error = "error"
          , Warning = "warning"
          , Info = "info"
          , Hint = "hint"
          }
          s

{- NodeMatcherF: The recursion scheme for NodeMatcher.
   
   This is a non-recursive type constructor that describes the structure
   of a NodeMatcher. The recursive field `has` uses a type parameter `r`
   instead of referring to the type itself.
   
   This is Step 1 of the Church encoding recipe.
-}
let NodeMatcherF = λ(r : Type) →
      { kind : Optional Text
      , field : Optional Text
      , regex : Optional Text
      , has : Optional r
      , inside : Optional r
      }

{- NodeMatcher: The Church-encoded recursive type.
   
   Step 2 of the Church encoding recipe:
   A value of type NodeMatcher is a function that, given:
   - A result type `r`
   - A function that knows how to fold a NodeMatcherF r into r
   returns a value of type r.
   
   This allows us to represent arbitrarily nested matchers without
   direct recursion in the type definition.
-}
let NodeMatcher = ∀(r : Type) → (NodeMatcherF r → r) → r

{- foldNodeMatcher: The fold function for NodeMatcher.
   
   This is the fundamental operation for consuming a NodeMatcher value.
   Given a result type `r` and a folding function, it applies the
   Church-encoded value to produce the result.
-}
let foldNodeMatcher : ∀(r : Type) → (NodeMatcherF r → r) → NodeMatcher → r =
      λ(r : Type) → λ(f : NodeMatcherF r → r) → λ(matcher : NodeMatcher) → matcher r f

{- nodeMatcher: Constructor for NodeMatcher values.
   
   Creates a NodeMatcher from a NodeMatcherF NodeMatcher record.
   This is the "build" function for the Church encoding.
-}
let nodeMatcher : NodeMatcherF NodeMatcher → NodeMatcher =
      λ(fm : NodeMatcherF NodeMatcher) →
        λ(r : Type) → λ(f : NodeMatcherF r → r) →
          let mapOptional : ∀(a : Type) → ∀(b : Type) → (a → b) → Optional a → Optional b =
                λ(a : Type) → λ(b : Type) → λ(g : a → b) → λ(o : Optional a) →
                  Prelude.Optional.fold a o (Optional b) (λ(x : a) → Some (g x)) (None b)
          
          let fr : NodeMatcherF r =
                { kind = fm.kind
                , field = fm.field
                , regex = fm.regex
                , has = mapOptional NodeMatcher r (foldNodeMatcher r f) fm.has
                , inside = mapOptional NodeMatcher r (foldNodeMatcher r f) fm.inside
                }
          in  f fr

{- emptyNodeMatcherF: Default value for NodeMatcherF. -}
let emptyNodeMatcherF : ∀(r : Type) → NodeMatcherF r =
      λ(r : Type) →
        { kind = None Text
        , field = None Text
        , regex = None Text
        , has = None r
        , inside = None r
        }

{- emptyNodeMatcher: A NodeMatcher with all fields empty. -}
let emptyNodeMatcher : NodeMatcher =
      λ(r : Type) → λ(f : NodeMatcherF r → r) → f (emptyNodeMatcherF r)

{- Rule: The main rule definition type.
   
   This is a simple record type (not recursive), containing:
   - kind: The AST node kind to match (required)
   - regex: Optional regex pattern for the matched text
   - pattern: Optional pattern to match
   - has: Optional nested matcher (as Church-encoded NodeMatcher)
   - not: Optional negation wrapper with has/inside fields
-}
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

{- Base lint type with properly typed Rule field. -}
let Lint =
      { id : Text
      , language : Text
      , severity : Severity
      , rule : Rule.Type
      , message : Text
      , note : Text
      , tests : { valid : List Text, invalid : List Text }
      }

{- Helper to convert Optional to List for concatenation -}
let Optional/toList =
      λ(a : Type) → λ(o : Optional a) →
        Prelude.Optional.fold a o (List a) (λ(x : a) → [ x ]) ([] : List a)

{- Convert NodeMatcherF JSON to JSON object fields. -}
let nodeMatcherFToJSONFields : NodeMatcherF JSON.Type → List { mapKey : Text, mapValue : JSON.Type } =
      λ(fm : NodeMatcherF JSON.Type) →
        Prelude.List.concat
          { mapKey : Text, mapValue : JSON.Type }
          [ Optional/toList
              { mapKey : Text, mapValue : JSON.Type }
              (Prelude.Optional.map
                 Text
                 { mapKey : Text, mapValue : JSON.Type }
                 (λ(k : Text) → { mapKey = "kind", mapValue = JSON.string k })
                 fm.kind)
          , Optional/toList
              { mapKey : Text, mapValue : JSON.Type }
              (Prelude.Optional.map
                 Text
                 { mapKey : Text, mapValue : JSON.Type }
                 (λ(f : Text) → { mapKey = "field", mapValue = JSON.string f })
                 fm.field)
          , Optional/toList
              { mapKey : Text, mapValue : JSON.Type }
              (Prelude.Optional.map
                 Text
                 { mapKey : Text, mapValue : JSON.Type }
                 (λ(r : Text) → { mapKey = "regex", mapValue = JSON.string r })
                 fm.regex)
          , Optional/toList
              { mapKey : Text, mapValue : JSON.Type }
              (Prelude.Optional.map
                 JSON.Type
                 { mapKey : Text, mapValue : JSON.Type }
                 (λ(h : JSON.Type) → { mapKey = "has", mapValue = h })
                 fm.has)
          , Optional/toList
              { mapKey : Text, mapValue : JSON.Type }
              (Prelude.Optional.map
                 JSON.Type
                 { mapKey : Text, mapValue : JSON.Type }
                 (λ(i : JSON.Type) → { mapKey = "inside", mapValue = i })
                 fm.inside)
          ]

{- Convert NodeMatcher to JSON using fold.
   
   The folding function builds up the JSON object from the NodeMatcherF structure.
-}
let nodeMatcherToJSON : NodeMatcher → JSON.Type =
      foldNodeMatcher
        JSON.Type
        (λ(fm : NodeMatcherF JSON.Type) → JSON.object (nodeMatcherFToJSONFields fm))

{- Convert Rule to JSON. -}
let ruleToJSON : Rule.Type → JSON.Type =
      λ(rule : Rule.Type) →
        let notFields =
              λ(n : { has : Optional NodeMatcher, inside : Optional NodeMatcher }) →
                Prelude.List.concat
                  { mapKey : Text, mapValue : JSON.Type }
                  [ Optional/toList
                      { mapKey : Text, mapValue : JSON.Type }
                      (Prelude.Optional.map
                         NodeMatcher
                         { mapKey : Text, mapValue : JSON.Type }
                         (λ(h : NodeMatcher) → { mapKey = "has", mapValue = nodeMatcherToJSON h })
                         n.has)
                  , Optional/toList
                      { mapKey : Text, mapValue : JSON.Type }
                      (Prelude.Optional.map
                         NodeMatcher
                         { mapKey : Text, mapValue : JSON.Type }
                         (λ(i : NodeMatcher) → { mapKey = "inside", mapValue = nodeMatcherToJSON i })
                         n.inside)
                  ]
        
        let fields =
              Prelude.List.concat
                { mapKey : Text, mapValue : JSON.Type }
                [ [ { mapKey = "kind", mapValue = JSON.string rule.kind } ]
                , Optional/toList
                    { mapKey : Text, mapValue : JSON.Type }
                    (Prelude.Optional.map
                       Text
                       { mapKey : Text, mapValue : JSON.Type }
                       (λ(r : Text) → { mapKey = "regex", mapValue = JSON.string r })
                       rule.regex)
                , Optional/toList
                    { mapKey : Text, mapValue : JSON.Type }
                    (Prelude.Optional.map
                       Text
                       { mapKey : Text, mapValue : JSON.Type }
                       (λ(p : Text) → { mapKey = "pattern", mapValue = JSON.string p })
                       rule.pattern)
                , Optional/toList
                    { mapKey : Text, mapValue : JSON.Type }
                    (Prelude.Optional.map
                       NodeMatcher
                       { mapKey : Text, mapValue : JSON.Type }
                       (λ(h : NodeMatcher) → { mapKey = "has", mapValue = nodeMatcherToJSON h })
                       rule.has)
                , Optional/toList
                    { mapKey : Text, mapValue : JSON.Type }
                    (Prelude.Optional.map
                       { has : Optional NodeMatcher, inside : Optional NodeMatcher }
                       { mapKey : Text, mapValue : JSON.Type }
                       (λ(n : { has : Optional NodeMatcher, inside : Optional NodeMatcher }) →
                         { mapKey = "not", mapValue = JSON.object (notFields n) })
                       rule.not)
                ]
        in  JSON.object fields

{- Render lint rule to YAML for directory tree. -}
let renderRuleYAML : Lint → Text =
      λ(lint : Lint) →
        let rule-json =
              JSON.object
                ( toMap
                    { id = JSON.string lint.id
                    , language = JSON.string lint.language
                    , severity = JSON.string (severityToText lint.severity)
                    , message = JSON.string lint.message
                    , note = JSON.string lint.note
                    , rule = ruleToJSON lint.rule
                    }
                )
        in  JSON.renderYAML rule-json

{- Render lint test to YAML for directory tree. -}
let renderTestYAML : Lint → Text =
      λ(lint : Lint) →
        let valid = Prelude.List.map Text JSON.Type JSON.string lint.tests.valid
        let invalid = Prelude.List.map Text JSON.Type JSON.string lint.tests.invalid
        let test-json =
              JSON.object
                ( toMap
                    { id = JSON.string lint.id
                    , valid = JSON.array valid
                    , invalid = JSON.array invalid
                    }
                )
        in  JSON.renderYAML test-json

{- Render sgconfig.yaml content. -}
let renderSGConfigYAML : Text =
      JSON.renderYAML
        ( JSON.object
            ( toMap
                { ruleDirs = JSON.array [ JSON.string "./rules" ]
                , testConfigs = JSON.array [ JSON.object (toMap { testDir = JSON.string "./tests" }) ]
                }
            )
        )

{- Export types and helpers. -}
in  { Severity = Severity
    , severityToText = severityToText
    , NodeMatcherF = NodeMatcherF
    , NodeMatcher = NodeMatcher
    , nodeMatcher = nodeMatcher
    , emptyNodeMatcher = emptyNodeMatcher
    , foldNodeMatcher = foldNodeMatcher
    , Rule = Rule
    , Lint = Lint
    , renderRuleYAML = renderRuleYAML
    , renderTestYAML = renderTestYAML
    , renderSGConfigYAML = renderSGConfigYAML
    }
