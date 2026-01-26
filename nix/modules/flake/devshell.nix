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
# Haskell packages: The build.nix module defines toolchain.haskell.packages
# as the single source of truth. Devshell adds testing/dev packages on top.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_:
{ config, lib, ... }:
let
  # lisp-case aliases for lib functions
  mk-enable-option = lib.mkEnableOption;
  mk-option = lib.mkOption;
  mk-if = lib.mkIf;
  optional-attrs = lib.optionalAttrs;
  optionals = lib.optionals;
  optional-string = lib.optionalString;
  has-prefix = lib.hasPrefix;
  filter = lib.filter;
  function-to = lib.types.functionTo;
  list-of = lib.types.listOf;
  attrs-of = lib.types.attrsOf;

  cfg = config.aleph.devshell;
  build-cfg = config.aleph.build;
in
{
  _class = "flake";

  options.aleph.devshell = {
    enable = mk-enable-option "aleph devshell";
    nv.enable = mk-enable-option "NVIDIA SDK in devshell";
    ghc-wasm.enable = mk-enable-option "GHC WASM backend (for builtins.wasm plugins)";
    straylight-nix.enable = mk-enable-option "straylight-nix with builtins.wasm support";

    extra-packages = mk-option {
      type = function-to (list-of lib.types.package);
      default = _: [ ];
      description = "Extra packages (receives pkgs)";
    };

    extra-haskell-packages = mk-option {
      type = function-to (list-of lib.types.package);
      default = hp: [
        # Scripting extras (not needed for Buck2 builds)
        hp.turtle
        hp.yaml
        hp.shelly
        hp.foldl
        hp.dhall
        hp.async

        # Crypto (for Aleph.Script.Oci etc)
        hp.crypton
        hp.memory

        # Data structures
        hp.unordered-containers
        hp.vector

        # Testing frameworks
        hp.hedgehog
        hp.QuickCheck
        hp.quickcheck-instances
        hp.tasty
        hp.tasty-quickcheck
        hp.tasty-hunit

        # Development utilities
        hp.lens
        hp.raw-strings-qq
      ];
      description = "Extra Haskell packages for devshell (on top of build.toolchain.haskell.packages)";
    };

    extra-shell-hook = mk-option {
      type = lib.types.lines;
      default = "";
      description = "Extra shell hook commands";
    };

    extra-env = mk-option {
      type = attrs-of lib.types.str;
      default = { };
      description = "Extra environment variables";
    };
  };

  config = mk-if cfg.enable {
    perSystem =
      { pkgs, config, ... }:
      let
        # All env vars defined here, not in shellHook
        # Env var names use CUDA/NVIDIA because that's what tools expect
        nv-env = optional-attrs (cfg.nv.enable && pkgs ? nvidia-sdk) {
          CUDA_HOME = "${pkgs.nvidia-sdk}";
          CUDA_PATH = "${pkgs.nvidia-sdk}";
          NVIDIA_SDK = "${pkgs.nvidia-sdk}";
          # LD_LIBRARY_PATH for runtime loading of CUDA libs (hasktorch, etc.)
          LD_LIBRARY_PATH = "${pkgs.nvidia-sdk}/lib";
        };

        # ────────────────────────────────────────────────────────────────────────
        # Haskell Configuration
        # ────────────────────────────────────────────────────────────────────────
        # Single source of truth: build.toolchain.haskell.packages from _main.nix
        # Devshell adds testing/dev packages on top via extra-haskell-packages.
        #
        # HLS go-to-definition:
        # - For YOUR code: works via hie.yaml (generated in shellHook)
        # - For library code: HLS uses Haddock docs for type info, but source
        #   navigation requires packages built with -fwrite-ide-info (not default).
        #   For library source nav, use haskell-src-exts or M-. in Emacs haskell-mode.
        hs-pkgs = pkgs.haskell.packages.ghc912 or pkgs.haskellPackages;

        # Combine build toolchain packages + devshell extras
        # build.toolchain.haskell.packages: core packages for Buck2 builds
        # cfg.extra-haskell-packages: testing, scripting, dev tools
        ghc-with-all-deps = hs-pkgs.ghcWithPackages (
          hp: (build-cfg.toolchain.haskell.packages hp) ++ (cfg.extra-haskell-packages hp)
        );

        # System libraries GHC needs at runtime
        ghc-runtime-libs = [
          pkgs.numactl
          pkgs.gmp
          pkgs.libffi
          pkgs.ncurses
        ];
      in
      {
        devShells.default = pkgs.mkShell (
          {
            name = "aleph-dev";

            hardeningDisable = [ "all" ];
            NIX_HARDENING_ENABLE = "";

            packages = [
              pkgs.git
              pkgs.jq
              pkgs.yq-go
              pkgs.ripgrep
              pkgs.fd
              pkgs.just
              pkgs.buck2
              ghc-with-all-deps

              # ════════════════════════════════════════════════════════════════
              # LSP servers - go-to-definition works out of the box
              # ════════════════════════════════════════════════════════════════

              # Haskell: HLS with matching GHC version
              # Packages built with -fwrite-ide-info for library navigation
              hs-pkgs.haskell-language-server

              # Nix: nixd (configured via .nixd.json from use_flake-lsp)
              pkgs.nixd

              # Rust: rust-analyzer (if Rust toolchain enabled)
              # Note: rust-analyzer is added via build.nix when rust toolchain is enabled
            ]
            # C++: clangd comes from llvm-git (via build.nix packages)
            # compile_commands.json generated by bin/compdb
            ++ ghc-runtime-libs
            ++ optionals (cfg.nv.enable && pkgs ? llvm-git) [
              pkgs.llvm-git
            ]
            ++ optionals (!cfg.nv.enable && pkgs ? aleph && pkgs.aleph ? llvm) [
              pkgs.aleph.llvm.clang
              pkgs.aleph.llvm.lld
            ]
            ++ optionals (cfg.nv.enable && pkgs ? nvidia-sdk) [
              pkgs.nvidia-sdk
            ]
            # GHC WASM toolchain for builtins.wasm plugin development
            ++ optionals (cfg.ghc-wasm.enable && pkgs ? aleph && pkgs.aleph ? ghc-wasm) (
              let
                ghc-wasm = pkgs.aleph.ghc-wasm;
              in
              filter (p: p != null) [
                ghc-wasm.ghc-wasm
                ghc-wasm.ghc-wasm-cabal
                ghc-wasm.wasi-sdk
                ghc-wasm.wasm-wasmtime
              ]
            )
            # straylight-nix with builtins.wasm support
            ++ optionals (cfg.straylight-nix.enable && pkgs ? aleph && pkgs.aleph ? nix) (
              filter (p: p != null) [
                pkgs.aleph.nix.nix
              ]
            )
            ++ (cfg.extra-packages pkgs)
            # Buck2 build system packages (excludes GHC since devshell has its own ghc-with-all-deps)
            # This includes llvm-git, nvidia-sdk, rustc, lean4, python, etc.
            ++ filter (p: !(has-prefix "ghc-" (p.name or ""))) (config.aleph.build.packages or [ ])
            # LRE packages (nativelink, lre-start)
            ++ (config.aleph.lre.packages or [ ]);

            shellHook =
              let
                ghc-wasm-check = optional-string cfg.ghc-wasm.enable ''
                  if command -v wasm32-wasi-ghc &>/dev/null; then
                    echo "GHC-WASM: $(wasm32-wasi-ghc --version)"
                  fi
                '';
                straylight-nix-check = optional-string cfg.straylight-nix.enable ''
                  if [ -n "${pkgs.aleph.nix.nix or ""}" ]; then
                    echo "straylight-nix: $(${pkgs.aleph.nix.nix}/bin/nix --version)"
                    echo "builtins.wasm: $(${pkgs.aleph.nix.nix}/bin/nix eval --expr 'builtins ? wasm')"
                  fi
                '';
              in
              ''
                echo "━━━ aleph devshell ━━━"
                echo "GHC: $(${ghc-with-all-deps}/bin/ghc --version)"
                ${ghc-wasm-check}
                ${straylight-nix-check}
                ${config.aleph.build.shellHook or ""}
                ${config.aleph.shortlist.shellHook or ""}
                ${config.aleph.lre.shellHook or ""}
                ${cfg.extra-shell-hook}
              '';
          }
          // nv-env
          // cfg.extra-env
        );
        devShells.linter = pkgs.mkShell {
          name = "linter-shell";

          packages = [
            pkgs.ast-grep
            pkgs.tree-sitter
            pkgs.tree-sitter-grammars.tree-sitter-nix
          ];
        };
      };
  };
}
