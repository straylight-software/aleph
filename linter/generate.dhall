{- Generate ast-grep configuration directory tree using dhall to-directory-tree

   This script generates the ast-grep configuration files:
   - sgconfig.yaml - The main configuration
   - rules/*.yml   - Individual rule files
   - tests/*.yml   - Test case files
   
   Usage in nix:
       dhall to-directory-tree --file ./generate.dhall --output ./ast-grep-config/
   
   See: https://hackage.haskell.org/package/dhall-1.41.2/docs/Dhall-DirectoryTree.html
-}

let Schema = ./schemas/Lint.dhall

let Prelude =
      https://prelude.dhall-lang.org/v20.1.0/package.dhall
        sha256:26b0ef498663d269e4dc6a82b0ee289ec565d683ef4c00d0ebdd25333a5a3c98

let Lint = Schema.Lint

-- All lint definitions
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

-- Generate rule file entries from lints
let ruleEntries =
      Prelude.List.map
        Lint
        { mapKey : Text, mapValue : Text }
        (λ(lint : Lint) → { mapKey = lint.id ++ ".yml", mapValue = Schema.renderRuleYAML lint })
        lints

-- Generate test file entries from lints
let testEntries =
      Prelude.List.map
        Lint
        { mapKey : Text, mapValue : Text }
        (λ(lint : Lint) → { mapKey = lint.id ++ ".yml", mapValue = Schema.renderTestYAML lint })
        lints

-- Build directory tree
in  { `sgconfig.yaml` = Schema.renderSGConfigYAML
    , `rules` = Prelude.List.fold
          { mapKey : Text, mapValue : Text }
          ruleEntries
          (Prelude.Map.Type Text Text)
          (λ(entry : { mapKey : Text, mapValue : Text }) → λ(acc : Prelude.Map.Type Text Text) → acc # [ entry ])
          ([] : Prelude.Map.Type Text Text)
    , `tests` = Prelude.List.fold
          { mapKey : Text, mapValue : Text }
          testEntries
          (Prelude.Map.Type Text Text)
          (λ(entry : { mapKey : Text, mapValue : Text }) → λ(acc : Prelude.Map.Type Text Text) → acc # [ entry ])
          ([] : Prelude.Map.Type Text Text)
    }
