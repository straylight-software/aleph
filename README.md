# // straylight // aleph //

> *system fomega for the sprawl*

______________________________________________________________________

infrastructure for modern computing. typed unix. static linking. system fomega, cic when it matters. droids ship code that works, often on the first try. when they're stuck, your codebase freezes rather than descending into madness. the ci knows exactly when things went off the rails, you back up, assess the wrong turn taken, and engage again.

this is not an easier way to develop software. it's a dramatically harder way. you make a critical executive decision every ten seconds for ten hours straight and when you're needed in the editor it's a welcome break from directing air traffic in contested airspace. you can't do it the same number of hours in a row, but that's ok because the improvement in outcomes grows faster than the increase in effort.

mathematics is, as usual, the key. we formalize everything we can, never satisfied by the imperfect confidence we get but it's still better this way. compiling one correct artifact into another correct artifact by a correct transformation is our default abstraction. system f suits human programmers, droids can't cope. system fomega it is. haskell and purescript are pretty nice actually.

![lambda cube showing system aleph-naught below untyped lambda calculus](docs/lambda-hierarchy.svg)

______________________________________________________________________

## // structure //

```
nix/
├── prelude/             # 100+ functions: map, filter, fold, maybe, either
├── overlays/
│   ├── libmodern/       # static c++: fmt, abseil, libsodium
│   ├── container/       # oci, firecracker, vfio
│   └── nvidia-sdk/      # cuda 13, cudnn, tensorrt
├── scripts/
│   └── Aleph/
│       ├── Script/      # typed unix: 25 cli wrappers, 32 compiled scripts
│       └── Nix/         # haskell dsl for derivations
└── modules/flake/       # drop-in flake configuration
```

______________________________________________________________________

## // quick start //

```nix
{
  inputs.aleph.url = "github:straylight-software/nix";

  outputs = { aleph, ... }:
    aleph.lib.mkFlake {
      perSystem = { prelude, pkgs, ... }: {
        devShells.default = prelude.mk-shell {

          deps = [ 
            pkgs.libmodern.abseil-cpp 
            pkgs.libmodern.fmt 
            pkgs.libmodern.libsodium 
          ];

          init-script = prelude.dev-init;
        };
      };
    };
}
```

______________________________________________________________________

## // typed unix //

replace bash with compiled haskell. ~2ms startup.

```haskell
#!/usr/bin/env runghc
import Aleph.Script
import qualified Aleph.Script.Tools.Rg as Rg

main :: IO ()
main = script $ do
  matches <- Rg.search "TODO" (Rg.defaults { Rg.glob = Just "*.hs" })
  mapM_ (echo . Rg.formatMatch) matches
```

25 tool wrappers. 32 scripts. zero bash.

______________________________________________________________________

## // prelude //

lisp-case everywhere. single obvious names. no abbreviations.

```nix
{ prelude, ... }:
let
  inherit (prelude) id const flip compose pipe;
  inherit (prelude) map filter fold head tail;
  inherit (prelude) maybe from-maybe is-just;
  inherit (prelude) left right either;
in { }
```

______________________________________________________________________

## // templates //

```bash
nix flake init -t github:straylight-software/nix          # standard
nix flake init -t github:straylight-software/nix#nv       # nvidia/ml
nix flake init -t github:straylight-software/nix#minimal  # just config
```

______________________________________________________________________

## // rfc //

| rfc | status |
|-----|--------|
| [001](docs/rfc/aleph-001-standard-nix.md) standard nix | implemented |
| [002](docs/rfc/aleph-002-lint.md) linting | implemented |
| [003](docs/rfc/aleph-003-prelude.md) the prelude | implemented |
| [004](docs/rfc/aleph-004-typed-unix.md) typed unix | implemented |
| [005](docs/rfc/aleph-005-profiles.md) nix profiles | implemented |
| [006](docs/rfc/aleph-006-safe-bash.md) safe bash | implemented |
| [007](docs/rfc/aleph-007-formalization.md) nix formalization | draft |

______________________________________________________________________

mit

______________________________________________________________________

*the smallest infinity is still infinite*
