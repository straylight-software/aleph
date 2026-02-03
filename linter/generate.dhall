{-
Generate ast-grep configuration directory tree.

Produces the following structure via `dhall to-directory-tree`:
- sgconfig.yaml  - Main ast-grep configuration
- rules/*.yml    - Individual rule definitions
- tests/*.yml    - Test case files

Usage:
    dhall to-directory-tree --file ./generate.dhall --output ./ast-grep-config/
-}
let Schema = ./schemas/Lint.dhall

let Prelude =
      https://prelude.dhall-lang.org/v20.1.0/package.dhall
        sha256:26b0ef498663d269e4dc6a82b0ee289ec565d683ef4c00d0ebdd25333a5a3c98

let Lint = Schema.Lint

let Entry = { mapKey : Text, mapValue : Text }

let lints
    : List Lint
    = [ ./lints/default-nix-in-packages.dhall
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

let renderToEntry
    : (Lint → Text) → Lint → Entry
    = λ(render : Lint → Text) →
      λ(lint : Lint) →
        { mapKey = lint.id ++ ".yml", mapValue = render lint }

let foldEntries
    : List Entry → Prelude.Map.Type Text Text
    = λ(entries : List Entry) →
        Prelude.List.fold
          Entry
          entries
          (Prelude.Map.Type Text Text)
          (λ(e : Entry) → λ(acc : Prelude.Map.Type Text Text) → acc # [ e ])
          ([] : Prelude.Map.Type Text Text)

in  { `sgconfig.yml` = Schema.renderSGConfigYAML
    , rules =
        foldEntries
          ( Prelude.List.map
              Lint
              Entry
              (renderToEntry Schema.renderRuleYAML)
              lints
          )
    , rule-tests =
        foldEntries
          ( Prelude.List.map
              Lint
              Entry
              (renderToEntry Schema.renderTestYAML)
              lints
          )
    }
