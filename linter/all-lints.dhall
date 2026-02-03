{- Main entry point for generating all linter configurations

   Usage:
       dhall-to-yaml-ng --documents --file ./all-lints.dhall
   
   This outputs all rules and test cases as separate YAML documents
   in the order: rule1, test1, rule2, test2, ...
-}

let Schema = ./schemas/Lint.dhall

let Lint = { id : Text
           , language : Text
           , severity : Schema.Severity
           , rule : {}
           , message : Text
           , note : Text
           , tests : { valid : List Text, invalid : List Text }
           }

-- Assert that a list is non-empty
let assertNonEmpty = λ(name : Text) → λ(list : List Text) →
      if Natural/isZero (List/length Text list)
      then merge { Some = λ(_ : Text) → _, None = "" } (None Text)
      else list

-- Process a single lint file and return both rule and test
let processLint = λ(lint : Lint) →
      let _ = assertNonEmpty "tests.valid" lint.tests.valid
      let _ = assertNonEmpty "tests.invalid" lint.tests.invalid
      in  [ Schema.toRule lint, Schema.toTestCase lint ]

-- Import all lint definitions
let lints =
      [ ./lints/default-nix-in-packages.dhall
      , ./lints/long-inline-string.dhall
      , ./lints/missing-class.dhall
      , ./lints/missing-description.dhall
      , ./lints/missing-meta.dhall
      , ./lints/no-heredoc-in-inline-bash.dhall
      , ./lints/non-lisp-case.dhall
      , ./lints/no-raw-mkderivation.dhall
      , ./lints/no-raw-runcommand.dhall
      , ./lints/no-raw-writeshellapplication.dhall
      , ./lints/no-substitute-all.dhall
      , ./lints/no-translate-attrs-outside-prelude.dhall
      , ./lints/or-null-fallback.dhall
      , ./lints/prefer-write-shell-application.dhall
      , ./lints/rec-anywhere.dhall
      , ./lints/rec-in-derivation.dhall
      , ./lints/with-lib.dhall
      ]

-- Process all lints and flatten into a single list of documents
in  List/concat
      { index : Natural, value : {} }
      (List/map Lint (List {}) processLint lints)
