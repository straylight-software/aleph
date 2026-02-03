{-
Generate ast-grep configuration directory tree.

Produces the following structure via `dhall to-directory-tree`:
- sgconfig.yml  - Main ast-grep configuration
- rules/*.yml    - Individual rule definitions
- rule-tests/*.yml    - Test case files

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
    = [ ./lints/nix/default-nix-in-packages.dhall
      , ./lints/nix/long-inline-string.dhall
      , ./lints/nix/missing-class.dhall
      , ./lints/nix/missing-description.dhall
      , ./lints/nix/missing-meta.dhall
      , ./lints/nix/no-heredoc-in-inline-bash.dhall
      , ./lints/nix/non-lisp-case.dhall
      , ./lints/nix/no-raw-mkderivation.dhall
      , ./lints/nix/no-raw-runcommand.dhall
      , ./lints/nix/no-raw-writeshellapplication.dhall
      , ./lints/nix/no-substitute-all.dhall
      , ./lints/nix/no-translate-attrs-outside-prelude.dhall
      , ./lints/nix/or-null-fallback.dhall
      , ./lints/nix/prefer-write-shell-application.dhall
      , ./lints/nix/rec-anywhere.dhall
      , ./lints/nix/rec-in-derivation.dhall
      , ./lints/nix/with-lib.dhall
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

let validateTestCount
    : Lint → Bool
    = λ(lint : Lint) →
        let validCount = Prelude.List.length Text lint.tests.valid

        let invalidCount = Prelude.List.length Text lint.tests.invalid

        in      Prelude.Natural.greaterThanEqual validCount 3
            &&  Prelude.Natural.greaterThanEqual invalidCount 3

let allTestsHaveAtLeast3Values
    : Bool
    = Prelude.List.all Lint validateTestCount lints

let _ = assert : allTestsHaveAtLeast3Values ≡ True

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
