{- Generate ast-grep configuration directory tree metadata

   This file generates metadata used by the nix build to create the directory tree.
   The actual YAML generation is done by nix using dhall-to-yaml-ng.
   
   See: https://hackage.haskell.org/package/dhall-1.41.2/docs/Dhall-DirectoryTree.html
-}

-- List of all lint IDs
[ "default-nix-in-packages"
, "long-inline-string"
, "missing-class"
, "missing-description"
, "missing-meta"
, "no-heredoc-in-inline-bash"
, "non-lisp-case"
, "no-raw-mkderivation"
, "no-raw-runcommand"
, "no-raw-writeshellapplication"
, "no-substitute-all"
, "no-translate-attrs-outside-prelude"
, "or-null-fallback"
, "prefer-write-shell-application"
, "rec-anywhere"
, "rec-in-derivation"
, "with-lib"
]
