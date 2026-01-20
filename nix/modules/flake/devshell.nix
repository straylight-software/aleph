# nix/modules/flake/devshell.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // devshell //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     Case had always taken it for granted that the real bosses, the
#     kingpins in a given industry, would be old men. The Tessier-
#     Ashpools were old money. He'd expected a boardroom, an
#     executive's office. Not the surreal maze of Straylight.
#
#                                                         — Neuromancer
#
# Development shell. Env vars at mkShell level only, not in shellHook.
# We say "nv" not "cuda". See: docs/languages/nix/philosophy/nvidia-not-cuda.md
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_:
{ config, lib, ... }:
let
  cfg = config.aleph-naught.devshell;
in
{
  _class = "flake";

  options.aleph-naught.devshell = {
    enable = lib.mkEnableOption "aleph-naught devshell";
    nv.enable = lib.mkEnableOption "NVIDIA SDK in devshell";
    ghc-wasm.enable = lib.mkEnableOption "GHC WASM backend (for builtins.wasm plugins)";
    straylight-nix.enable = lib.mkEnableOption "straylight-nix with builtins.wasm support";

    extra-packages = lib.mkOption {
      type = lib.types.functionTo (lib.types.listOf lib.types.package);
      default = _: [ ];
      description = "Extra packages (receives pkgs)";
    };

    extra-shell-hook = lib.mkOption {
      type = lib.types.lines;
      default = "";
      description = "Extra shell hook commands";
    };

    extra-env = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Extra environment variables";
    };
  };

  config = lib.mkIf cfg.enable {
    perSystem =
      { pkgs, config, ... }:
      let
        # All env vars defined here, not in shellHook
        # Env var names use CUDA/NVIDIA because that's what tools expect
        nv-env = lib.optionalAttrs (cfg.nv.enable && pkgs ? nvidia-sdk) {
          CUDA_HOME = "${pkgs.nvidia-sdk}";
          CUDA_PATH = "${pkgs.nvidia-sdk}";
          NVIDIA_SDK = "${pkgs.nvidia-sdk}";
        };

        # Mercury GHC and package set from prelude
        inherit (config.straylight) prelude;
        mercuryGhc = prelude.ghc.pkg;
        mercuryHsPkgs = prelude.ghc.pkgs';

        # System libraries Mercury GHC needs
        mercuryGhcLibs = [
          pkgs.numactl
          pkgs.gmp
          pkgs.libffi
          pkgs.ncurses
        ];

        # Haskell packages for development
        # TODO: Make this configurable via module options instead of hardcoding
        # Currently includes zeitschrift deps for bootstrap - remove once zeitschrift
        # has its own package management
        haskellDeps = hp: [
          # Core / scripting
          hp.megaparsec
          hp.text
          hp.bytestring
          hp.containers
          hp.aeson
          hp.optparse-applicative
          hp.turtle
          hp.hedgehog

          # zeitschrift deps (temporary - for bootstrap)
          hp.yaml
          hp.prettyprinter
          hp.openapi3
          hp.lens
          hp.insert-ordered-containers
          hp.lucid
          hp.http-media
          hp.QuickCheck
          hp.directory
          hp.filepath
          hp.time
          hp.process
          hp.quickcheck-instances
          hp.tasty
          hp.tasty-quickcheck
          hp.tasty-hunit
          hp.raw-strings-qq
        ];

        # Individual package derivations
        haskellPkgs = haskellDeps mercuryHsPkgs;
        haskellPkgsPaths = lib.concatMapStringsSep " " toString haskellPkgs;
      in
      {
        devShells.default = pkgs.mkShell (
          {
            name = "straylight-dev";

            hardeningDisable = [ "all" ];
            NIX_HARDENING_ENABLE = "";

            packages = [
              pkgs.git
              pkgs.jq
              pkgs.yq-go
              pkgs.ripgrep
              pkgs.fd
              pkgs.just
              mercuryGhc
            ]
            ++ haskellPkgs
            ++ mercuryGhcLibs
            ++ lib.optionals (cfg.nv.enable && pkgs ? llvm-git) [
              pkgs.llvm-git
            ]
            ++ lib.optionals (!cfg.nv.enable && pkgs ? straylight && pkgs.straylight ? llvm) [
              pkgs.straylight.llvm.clang
              pkgs.straylight.llvm.lld
            ]
            ++ lib.optionals (cfg.nv.enable && pkgs ? nvidia-sdk) [
              pkgs.nvidia-sdk
            ]
            # GHC WASM toolchain for builtins.wasm plugin development
            ++ lib.optionals (cfg.ghc-wasm.enable && pkgs ? straylight && pkgs.straylight ? ghc-wasm) (
              let
                ghcWasm = pkgs.straylight.ghc-wasm;
              in
              lib.filter (p: p != null) [
                ghcWasm.ghc-wasm
                ghcWasm.ghc-wasm-cabal
                ghcWasm.wasi-sdk
                ghcWasm.wasm-wasmtime
              ]
            )
            # straylight-nix with builtins.wasm support
            ++ lib.optionals (cfg.straylight-nix.enable && pkgs ? straylight && pkgs.straylight ? straylight-nix) (
              lib.filter (p: p != null) [
                pkgs.straylight.nix.straylight-nix
              ]
            )
            ++ (cfg.extra-packages pkgs);

            shellHook = ''
              echo "━━━ aleph-naught devshell ━━━"

              # Create combined package database for Mercury GHC
              PKGDB_DIR="$PWD/.ghc-pkg-db"
              if [ ! -f "$PKGDB_DIR/package.cache" ]; then
                echo "Creating GHC package database..."
                rm -rf "$PKGDB_DIR"
                mkdir -p "$PKGDB_DIR"

                # Collect .conf files from Haskell packages
                collect_confs() {
                  local pkg="$1"
                  local visited="$2"
                  if echo "$visited" | grep -q "$pkg"; then return; fi
                  visited="$visited $pkg"
                  for confdir in "$pkg"/lib/ghc-*/package.conf.d; do
                    if [ -d "$confdir" ]; then
                      cp "$confdir"/*.conf "$PKGDB_DIR/" 2>/dev/null || true
                    fi
                  done
                  if [ -f "$pkg/nix-support/propagated-build-inputs" ]; then
                    for dep in $(cat "$pkg/nix-support/propagated-build-inputs"); do
                      collect_confs "$dep" "$visited"
                    done
                  fi
                }

                VISITED=""
                for pkg in ${haskellPkgsPaths}; do
                  collect_confs "$pkg" "$VISITED"
                done

                ${mercuryGhc}/bin/ghc-pkg --package-db="$PKGDB_DIR" recache
                echo "Package database: $(ls "$PKGDB_DIR"/*.conf 2>/dev/null | wc -l) packages"
              fi

              export GHC_PACKAGE_PATH="$PKGDB_DIR:"
              echo "GHC: $(${mercuryGhc}/bin/ghc --version)"
              ${lib.optionalString cfg.ghc-wasm.enable ''
                if command -v wasm32-wasi-ghc &>/dev/null; then
                  echo "GHC-WASM: $(wasm32-wasi-ghc --version)"
                fi
              ''}
              ${lib.optionalString cfg.straylight-nix.enable ''
                if [ -n "${pkgs.straylight.nix.straylight-nix or ""}" ]; then
                  echo "straylight-nix: $(${pkgs.straylight.nix.straylight-nix}/bin/nix --version)"
                  echo "builtins.wasm: $(${pkgs.straylight.nix.straylight-nix}/bin/nix eval --expr 'builtins ? wasm')"
                fi
              ''}
              ${cfg.extra-shell-hook}
            '';
          }
          // nv-env
          // cfg.extra-env
        );
      };
  };
}
