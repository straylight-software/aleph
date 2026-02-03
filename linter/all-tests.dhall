{- Generate all lint test cases for ast-grep

   Usage:
       dhall-to-yaml-ng --documents --file ./all-tests.dhall
   
   Outputs all test definitions as separate YAML documents
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

-- Assert that a list is non-empty (fails build if tests are empty)
let assertNonEmpty = λ(name : Text) → λ(list : List Text) →
      if Natural/isZero (List/length Text list)
      then merge { Some = λ(_ : Text) → _, None = "" } (None Text)
      else list

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

-- Convert all lints to test cases with validation
let toTest = λ(lint : Lint) →
      let _ = assertNonEmpty "tests.valid" lint.tests.valid
      let _ = assertNonEmpty "tests.invalid" lint.tests.invalid
      in  Schema.toTestCase lint

in  List/map Lint {} toTest lints
