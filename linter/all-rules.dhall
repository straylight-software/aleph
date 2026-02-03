{- Generate all lint rules for ast-grep

   Usage:
       dhall-to-yaml-ng --documents --file ./all-rules.dhall
   
   Outputs all rule definitions as separate YAML documents
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

-- Convert all lints to rules
in  List/map Lint {} Schema.toRule lints
