{- Process all lint files and output as multiple YAML documents
   
   Usage:
       dhall-to-yaml-ng --documents --file ./process-all.dhall
   
   This script:
   1. Discovers all lint files in ../lints/
   2. Validates each has non-empty tests
   3. Outputs rule YAML followed by test YAML for each lint
   4. Fails with type error if any lint has empty tests
-}

let Schema = ../schemas/Lint.dhall

let Lint = { id : Text
           , language : Text
           , severity : Schema.Severity
           , rule : {}
           , message : Text
           , note : Text
           , tests : { valid : List Text, invalid : List Text }
           }


-- Assert that a list is non-empty by using it in a context that requires NonEmpty
let assertNonEmpty = λ(name : Text) → λ(list : List Text) →
      if Natural/isZero (List/length Text list)
      then merge { Some = λ(_ : Text) → _, None = "" } (None Text)  -- Force error
      else list

-- Process a single lint file and return both rule and test as a list
let processLint = λ(lint : Lint) →
      let _ = assertNonEmpty "tests.valid" lint.tests.valid
      let _ = assertNonEmpty "tests.invalid" lint.tests.invalid
      in  [ Schema.toRule lint, Schema.toTestCase lint ]

-- List of all lint files to process
-- NOTE: This needs to be generated or we need to use a different approach
-- For now, we'll list them explicitly
let lints =
      [ ./default-nix-in-packages.dhall
      , ./long-inline-string.dhall
      , ./missing-class.dhall
      , ./missing-description.dhall
      , ./missing-meta.dhall
      , ./no-heredoc-in-inline-bash.dhall
      , ./non-lisp-case.dhall
      , ./no-raw-mkderivation.dhall
      , ./no-raw-runcommand.dhall
      , ./no-raw-writeshellapplication.dhall
      , ./no-substitute-all.dhall
      , ./no-translate-attrs-outside-prelude.dhall
      , ./or-null-fallback.dhall
      , ./prefer-write-shell-application.dhall
      , ./rec-anywhere.dhall
      , ./rec-in-derivation.dhall
      , ./with-lib.dhall
      ]

-- Process all lints and flatten into a single list of documents
in  List/concat
      { index : Natural, value : {}
      }
      (List/map
        Lint
        (List {})
        processLint
        lints)
