# nix/modules/flake/build.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // build //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The matrix has its roots in primitive arcade games, in early
#     graphics programs and military experimentation with cranial
#     jacks. On the Sony, a two-dimensional space war faded behind
#     a forest of mathematically generated ferns, demonstrating the
#     spatial possibilities of logarithmic spirals.
#
#                                                         — Neuromancer
#
# Buck2 build system integration. Provides:
#   - Hermetic LLVM 22 toolchain paths for .buckconfig.local
#   - Buck2 prelude (straylight fork with NVIDIA support)
#   - Toolchain definitions (.bzl files)
#   - Toolchain wrappers (GHC, Lean, Python/nanobind)
#
# USAGE (downstream flake):
#
#   {
#     inputs.aleph.url = "github:straylight-software/aleph";
#     inputs.buck2-prelude.url = "github:weyl-ai/straylight-buck2-prelude";
#     inputs.buck2-prelude.flake = false;
#
#     outputs = { self, aleph, buck2-prelude, ... }:
#       aleph.lib.mkFlake { inherit inputs; } {
#         imports = [ aleph.modules.flake.build ];
#
#         aleph-naught.build = {
#           enable = true;
#           toolchain.cxx.enable = true;
#           toolchain.nv.enable = true;
#         };
#       };
#   }
#
# This generates:
#   - .buckconfig.local with Nix store paths (via devshell hook)
#   - nix/build/prelude symlink to buck2-prelude
#   - nix/build/toolchains with .bzl files
#
# We say "nv" not "cuda". We use clang for .cu files, not nvcc.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
{
  config,
  lib,
  flake-parts-lib,
  ...
}:
let
  inherit (flake-parts-lib) mkPerSystemOption;
  cfg = config.aleph-naught.build;
in
{
  _class = "flake";

  # ════════════════════════════════════════════════════════════════════════════
  # Per-system options for straylight.build
  # ════════════════════════════════════════════════════════════════════════════
  options.perSystem = mkPerSystemOption (
    { lib, ... }:
    {
      options.straylight.build = {
        buck2-toolchain = lib.mkOption {
          type = lib.types.raw;
          default = { };
          description = "Buck2 toolchain paths from .buckconfig.local";
        };
        buckconfig-local = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          description = "Path to generated .buckconfig.local";
        };
        shellHook = lib.mkOption {
          type = lib.types.lines;
          default = "";
          description = "Shell hook for Buck2 setup";
        };
        packages = lib.mkOption {
          type = lib.types.listOf lib.types.package;
          default = [ ];
          description = "Packages for Buck2 toolchains";
        };
      };
    }
  );

  options.aleph-naught.build = {
    enable = lib.mkEnableOption "Buck2 build system integration";

    # ──────────────────────────────────────────────────────────────────────────
    # Prelude Configuration
    # ──────────────────────────────────────────────────────────────────────────
    prelude = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Include straylight-buck2-prelude in flake outputs";
      };

      path = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = ''
          Path to buck2 prelude. If null, uses inputs.buck2-prelude.
          Set this to vendor a local copy.
        '';
      };
    };

    # ──────────────────────────────────────────────────────────────────────────
    # Toolchain Configuration
    # ──────────────────────────────────────────────────────────────────────────
    toolchain = {
      cxx = {
        enable = lib.mkEnableOption "C++ toolchain (LLVM 22)";

        c-flags = lib.mkOption {
          type = lib.types.listOf lib.types.str;
          default = [
            "-O2"
            "-g3"
            "-gdwarf-5"
            "-fno-omit-frame-pointer"
            "-mno-omit-leaf-frame-pointer"
            "-U_FORTIFY_SOURCE"
            "-D_FORTIFY_SOURCE=0"
            "-fno-stack-protector"
            "-std=c23"
            "-Wall"
            "-Wextra"
          ];
          description = "C compiler flags";
        };

        cxx-flags = lib.mkOption {
          type = lib.types.listOf lib.types.str;
          default = [
            "-O2"
            "-g3"
            "-gdwarf-5"
            "-fno-omit-frame-pointer"
            "-mno-omit-leaf-frame-pointer"
            "-U_FORTIFY_SOURCE"
            "-D_FORTIFY_SOURCE=0"
            "-fno-stack-protector"
            "-std=c++23"
            "-Wall"
            "-Wextra"
            "-fno-exceptions"
          ];
          description = "C++ compiler flags";
        };

        link-style = lib.mkOption {
          type = lib.types.enum [
            "static"
            "shared"
          ];
          default = "static";
          description = "Default link style";
        };
      };

      nv = {
        enable = lib.mkEnableOption "NVIDIA toolchain (clang + nvidia-sdk)";

        archs = lib.mkOption {
          type = lib.types.listOf lib.types.str;
          default = [
            "sm_90"
            "sm_100"
            "sm_120"
          ];
          description = ''
            Target NVIDIA architectures:
              sm_90  = Hopper (H100)
              sm_100 = Blackwell (B100, B200)
              sm_120 = Blackwell (B200 full features, requires LLVM 22)
          '';
        };
      };

      haskell = {
        enable = lib.mkEnableOption "Haskell toolchain (GHC from Nix)";

        packages = lib.mkOption {
          type = lib.types.functionTo (lib.types.listOf lib.types.package);
          default =
            hp:
            lib.filter (p: p != null) [
              hp.text or null
              hp.bytestring or null
              hp.containers or null
              hp.aeson or null
              hp.optparse-applicative or null
            ];
          description = "Haskell packages for Buck2 toolchain (receives haskellPackages)";
        };
      };

      rust = {
        enable = lib.mkEnableOption "Rust toolchain";
      };

      lean = {
        enable = lib.mkEnableOption "Lean 4 toolchain";
      };

      python = {
        enable = lib.mkEnableOption "Python toolchain (with nanobind)";

        packages = lib.mkOption {
          type = lib.types.functionTo (lib.types.listOf lib.types.package);
          default =
            ps:
            lib.filter (p: p != null) [
              ps.nanobind or null
              ps.numpy or null
            ];
          description = "Python packages for Buck2 toolchain";
        };
      };
    };

    # ──────────────────────────────────────────────────────────────────────────
    # Output Configuration
    # ──────────────────────────────────────────────────────────────────────────
    generate-buckconfig = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Generate .buckconfig.local in devshell";
    };

    generate-wrappers = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Generate bin/ wrappers for toolchains";
    };

    generate-buckconfig-main = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Generate .buckconfig if missing.
        Set to true for downstream projects that don't have their own .buckconfig.
      '';
    };

    # ──────────────────────────────────────────────────────────────────────────
    # IDE Integration
    # ──────────────────────────────────────────────────────────────────────────
    compdb = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Generate compile_commands.json for clangd/clang-tidy";
      };

      targets = lib.mkOption {
        type = lib.types.listOf lib.types.str;
        default = [ "//..." ];
        description = "Buck2 targets to include in compile_commands.json";
      };

      auto-generate = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = ''
          Auto-generate compile_commands.json on shell entry.
          Can be slow for large projects. Use bin/compdb manually instead.
        '';
      };
    };
  };

  config = lib.mkIf cfg.enable {
    # ══════════════════════════════════════════════════════════════════════════
    # Nixpkgs overlays - automatically add required overlays
    # ══════════════════════════════════════════════════════════════════════════
    aleph-naught.nixpkgs.overlays = lib.mkBefore [
      # LLVM 22 overlay (for llvm-git package)
      (import ../../overlays/llvm-git.nix inputs)
      # Packages overlay (for mdspan)
      (final: _prev: {
        mdspan = final.callPackage ../../overlays/packages/mdspan.nix { };
      })
      # NVIDIA SDK overlay
      (import ../../overlays/nixpkgs-nvidia-sdk.nix)
    ];

    # ══════════════════════════════════════════════════════════════════════════
    # Flake-level outputs
    # ══════════════════════════════════════════════════════════════════════════
    flake = {
      # Export the prelude for downstream consumers
      buck2-prelude = lib.mkIf cfg.prelude.enable (
        if cfg.prelude.path != null then cfg.prelude.path else inputs.buck2-prelude or null
      );

      # ────────────────────────────────────────────────────────────────────────
      # Export toolchains as a derivation
      # ────────────────────────────────────────────────────────────────────────
      # Downstream flakes can symlink this to their toolchains/ directory
      # or reference it via the `toolchains` cell in .buckconfig
      buck2-toolchains = inputs.self + "/toolchains";

      # ────────────────────────────────────────────────────────────────────────
      # Export .buckconfig template
      # ────────────────────────────────────────────────────────────────────────
      buck2-config-template = ''
        # Buck2 Configuration (generated by aleph.build)
        #
        # Hermetic builds with Nix store paths.
        # Toolchain paths are set in .buckconfig.local (generated by nix develop).

        [cells]
        root = .
        prelude = nix/build/prelude
        toolchains = nix/build/toolchains
        none = none

        [cell_aliases]
        config = prelude
        ovr_config = prelude
        fbcode = none
        fbsource = none
        fbcode_macros = none
        buck = none

        [parser]
        target_platform_detector_spec = target:root//...->prelude//platforms:default

        [buck2]
        materializations = deferred
        digest_algorithms = BLAKE3
      '';
    };

    # ══════════════════════════════════════════════════════════════════════════
    # Per-system configuration
    # ══════════════════════════════════════════════════════════════════════════
    perSystem =
      { pkgs, ... }:
      let
        # ────────────────────────────────────────────────────────────────────────
        # Toolchain paths (Linux only)
        # ────────────────────────────────────────────────────────────────────────
        inherit (pkgs.stdenv) isLinux;

        # LLVM 22 from llvm-git overlay
        llvm-git = pkgs.llvm-git or null;

        # GCC for libstdc++ headers and runtime
        gcc = pkgs.gcc15 or pkgs.gcc14 or pkgs.gcc;
        gcc-unwrapped = gcc.cc;
        gcc-version = gcc-unwrapped.version;
        triple = pkgs.stdenv.hostPlatform.config;

        # NVIDIA SDK
        nvidia-sdk = pkgs.nvidia-sdk or null;

        # mdspan (Kokkos reference implementation)
        mdspan = pkgs.mdspan or null;

        # Haskell
        hsPkgs = pkgs.haskell.packages.ghc912 or pkgs.haskellPackages;
        ghcForBuck2 = hsPkgs.ghcWithPackages cfg.toolchain.haskell.packages;

        # Python
        python = pkgs.python312 or pkgs.python311;
        pythonEnv = python.withPackages cfg.toolchain.python.packages;

        # ────────────────────────────────────────────────────────────────────────
        # Buck2 toolchain configuration
        # ────────────────────────────────────────────────────────────────────────
        buck2-toolchain =
          lib.optionalAttrs (isLinux && llvm-git != null) {
            # LLVM 22 compilers
            cc = "${llvm-git}/bin/clang";
            cxx = "${llvm-git}/bin/clang++";
            cpp = "${llvm-git}/bin/clang-cpp";

            # LLVM 22 bintools
            ar = "${llvm-git}/bin/llvm-ar";
            ld = "${llvm-git}/bin/ld.lld";
            nm = "${llvm-git}/bin/llvm-nm";
            objcopy = "${llvm-git}/bin/llvm-objcopy";
            objdump = "${llvm-git}/bin/llvm-objdump";
            strip = "${llvm-git}/bin/llvm-strip";
            ranlib = "${llvm-git}/bin/llvm-ranlib";

            # Include directories
            clang-resource-dir = "${llvm-git}/lib/clang/22";
            gcc-include = "${gcc-unwrapped}/include/c++/${gcc-version}";
            gcc-include-arch = "${gcc-unwrapped}/include/c++/${gcc-version}/${triple}";
            glibc-include = "${pkgs.glibc.dev}/include";

            # Library directories
            gcc-lib = "${gcc-unwrapped}/lib/gcc/${triple}/${gcc-version}";
            gcc-lib-base = "${gcc.cc.lib}/lib";
            glibc-lib = "${pkgs.glibc}/lib";
          }
          // lib.optionalAttrs (isLinux && mdspan != null) {
            mdspan-include = "${mdspan}/include";
          }
          // lib.optionalAttrs (isLinux && nvidia-sdk != null && cfg.toolchain.nv.enable) {
            nvidia-sdk-path = "${nvidia-sdk}";
            nvidia-sdk-include = "${nvidia-sdk}/include";
            nvidia-sdk-lib = "${nvidia-sdk}/lib";
          }
          // lib.optionalAttrs (isLinux && cfg.toolchain.python.enable) {
            python-interpreter = "${pythonEnv}/bin/python3";
            python-include = "${python}/include/python3.12";
            python-lib = "${python}/lib";
            nanobind-include = "${python.pkgs.nanobind}/lib/python3.12/site-packages/nanobind/include";
            nanobind-cmake = "${python.pkgs.nanobind}/lib/python3.12/site-packages/nanobind";
          };

        # ────────────────────────────────────────────────────────────────────────
        # .buckconfig.local generator
        # ────────────────────────────────────────────────────────────────────────
        buckconfig-local = pkgs.writeText "buckconfig.local" (
          lib.optionalString (cfg.toolchain.cxx.enable && buck2-toolchain ? cc) ''
            # AUTO-GENERATED by aleph.build module
            # DO NOT EDIT - regenerated on each shell entry

            [cxx]
            # Compilers (LLVM 22)
            cc = ${buck2-toolchain.cc}
            cxx = ${buck2-toolchain.cxx}
            cpp = ${buck2-toolchain.cpp}

            # Bintools
            ar = ${buck2-toolchain.ar}
            ld = ${buck2-toolchain.ld}

            # Include paths
            clang_resource_dir = ${buck2-toolchain.clang-resource-dir}
            gcc_include = ${buck2-toolchain.gcc-include}
            gcc_include_arch = ${buck2-toolchain.gcc-include-arch}
            glibc_include = ${buck2-toolchain.glibc-include}

            # Library paths
            gcc_lib = ${buck2-toolchain.gcc-lib}
            gcc_lib_base = ${buck2-toolchain.gcc-lib-base}
            glibc_lib = ${buck2-toolchain.glibc-lib}

            ${lib.optionalString (buck2-toolchain ? mdspan-include) ''
              # mdspan (Kokkos reference implementation)
              mdspan_include = ${buck2-toolchain.mdspan-include}
            ''}
          ''
          + lib.optionalString cfg.toolchain.haskell.enable ''

            [haskell]
            ghc = ${ghcForBuck2}/bin/ghc
            ghc_pkg = ${ghcForBuck2}/bin/ghc-pkg
            haddock = ${ghcForBuck2}/bin/haddock
            ghc_lib_dir = ${ghcForBuck2}/lib/ghc-${hsPkgs.ghc.version}/lib
            global_package_db = ${ghcForBuck2}/lib/ghc-${hsPkgs.ghc.version}/lib/package.conf.d
          ''
          + lib.optionalString (cfg.toolchain.rust.enable && pkgs ? rustc) ''

            [rust]
            rustc = ${pkgs.rustc}/bin/rustc
            rustdoc = ${pkgs.rustc}/bin/rustdoc
            clippy_driver = ${pkgs.clippy}/bin/clippy-driver
            cargo = ${pkgs.cargo}/bin/cargo
            target_triple = x86_64-unknown-linux-gnu
          ''
          + lib.optionalString (cfg.toolchain.lean.enable && pkgs ? lean4) ''

            [lean]
            lean = ${pkgs.lean4}/bin/lean
            leanc = ${pkgs.lean4}/bin/leanc
            lake = ${pkgs.lean4}/bin/lake
            lean_lib_dir = ${pkgs.lean4}/lib/lean/library
            lean_include_dir = ${pkgs.lean4}/include
          ''
          + lib.optionalString (cfg.toolchain.python.enable && buck2-toolchain ? python-interpreter) ''

            [python]
            interpreter = ${buck2-toolchain.python-interpreter}
            python_include = ${buck2-toolchain.python-include}
            python_lib = ${buck2-toolchain.python-lib}
            nanobind_include = ${buck2-toolchain.nanobind-include}
            nanobind_cmake = ${buck2-toolchain.nanobind-cmake}
          ''
          + lib.optionalString (cfg.toolchain.nv.enable && buck2-toolchain ? nvidia-sdk-path) ''

            [nv]
            nvidia_sdk_path = ${buck2-toolchain.nvidia-sdk-path}
            nvidia_sdk_include = ${buck2-toolchain.nvidia-sdk-include}
            nvidia_sdk_lib = ${buck2-toolchain.nvidia-sdk-lib}
          ''
        );

        # ────────────────────────────────────────────────────────────────────────
        # Shell hook for generating .buckconfig.local and wrappers
        # ────────────────────────────────────────────────────────────────────────
        # Buck2 prelude source
        preludeSrc = if cfg.prelude.path != null then cfg.prelude.path else inputs.buck2-prelude or null;

        # Toolchains source (from this flake)
        toolchainsSrc = inputs.self + "/toolchains";

        buildShellHook =
          lib.optionalString (isLinux && cfg.prelude.enable && preludeSrc != null) ''
            # Link buck2-prelude to nix/build/prelude
            if [ ! -e "nix/build/prelude/prelude.bzl" ]; then
              echo "Linking buck2-prelude..."
              rm -rf nix/build/prelude
              mkdir -p nix/build
              ln -sf ${preludeSrc} nix/build/prelude
              echo "Linked ${preludeSrc} → nix/build/prelude"
            fi

            # Link toolchains to nix/build/toolchains
            if [ ! -e "nix/build/toolchains/cxx.bzl" ]; then
              echo "Linking buck2-toolchains..."
              rm -rf nix/build/toolchains
              mkdir -p nix/build
              ln -sf ${toolchainsSrc} nix/build/toolchains
              echo "Linked ${toolchainsSrc} → nix/build/toolchains"
            fi
          ''
          + lib.optionalString (isLinux && cfg.generate-buckconfig-main) ''
                        # Generate .buckconfig if missing
                        if [ ! -e ".buckconfig" ]; then
                          echo "Generating .buckconfig..."
                          cat > .buckconfig << 'BUCKCONFIG_EOF'
            # Buck2 Configuration (generated by aleph.build)
            #
            # Hermetic builds with Nix store paths.
            # Toolchain paths are set in .buckconfig.local (generated by nix develop).

            [cells]
            root = .
            prelude = nix/build/prelude
            toolchains = nix/build/toolchains
            none = none

            [cell_aliases]
            config = prelude
            ovr_config = prelude
            fbcode = none
            fbsource = none
            fbcode_macros = none
            buck = none

            [parser]
            target_platform_detector_spec = target:root//...->prelude//platforms:default

            [buck2]
            materializations = deferred
            digest_algorithms = BLAKE3
            BUCKCONFIG_EOF
                          echo "Generated .buckconfig"
                        fi

                        # Generate .buckroot if missing
                        if [ ! -e ".buckroot" ]; then
                          touch .buckroot
                          echo "Generated .buckroot"
                        fi

                        # Generate none/BUCK if missing
                        if [ ! -e "none/BUCK" ]; then
                          mkdir -p none
                          touch none/BUCK
                          echo "Generated none/BUCK"
                        fi
          ''
          + lib.optionalString (isLinux && cfg.generate-buckconfig) ''
            # Generate .buckconfig.local with Nix store paths
            rm -f .buckconfig.local 2>/dev/null || true
            cp ${buckconfig-local} .buckconfig.local
            chmod 644 .buckconfig.local
            echo "Generated .buckconfig.local with Nix store paths"
          ''
          + lib.optionalString (isLinux && cfg.generate-wrappers) ''
            # Generate bin/ wrappers for Buck2 toolchains
            mkdir -p bin

            ${lib.optionalString cfg.toolchain.haskell.enable ''
                # GHC wrapper - filters Mercury-specific flags for stock GHC
                cat > bin/ghc << 'GHC_WRAPPER_EOF'
              #!/usr/bin/env bash
              # Buck2 GHC wrapper - filters Mercury-specific flags for stock GHC
              filter_args() {
                  local skip_next=false
                  while IFS= read -r arg || [[ -n "$arg" ]]; do
                      if $skip_next; then skip_next=false; continue; fi
                      case "$arg" in
                          -dep-json) skip_next=true ;;
                          -fpackage-db-byte-code|-fprefer-byte-code|-fbyte-code-and-object-code|-hide-all-packages) ;;
                          *) echo "$arg" ;;
                      esac
                  done
              }
              final_args=()
              for arg in "$@"; do
                  if [[ "$arg" == @* ]]; then
                      response_file="''${arg:1}"
                      if [[ -f "$response_file" ]]; then
                          filtered_file=$(mktemp)
                          filter_args < "$response_file" > "$filtered_file"
                          final_args+=("@$filtered_file")
                          trap "rm -f '$filtered_file'" EXIT
                      else
                          final_args+=("$arg")
                      fi
                  else
                      case "$arg" in
                          -dep-json|-fpackage-db-byte-code|-fprefer-byte-code|-fbyte-code-and-object-code|-hide-all-packages) ;;
                          *) final_args+=("$arg") ;;
                      esac
                  fi
              done
              exec ghc "''${final_args[@]}"
              GHC_WRAPPER_EOF
                chmod +x bin/ghc

                cat > bin/ghc-pkg << 'EOF'
              #!/usr/bin/env bash
              exec ghc-pkg "$@"
              EOF
                chmod +x bin/ghc-pkg

                cat > bin/haddock << 'EOF'
              #!/usr/bin/env bash
              exec haddock "$@"
              EOF
                chmod +x bin/haddock
            ''}

            ${lib.optionalString cfg.toolchain.lean.enable ''
                cat > bin/lean << 'EOF'
              #!/usr/bin/env bash
              exec lean "$@"
              EOF
                chmod +x bin/lean

                cat > bin/leanc << 'EOF'
              #!/usr/bin/env bash
              exec leanc "$@"
              EOF
                chmod +x bin/leanc
            ''}

            ${lib.optionalString cfg.toolchain.cxx.enable ''
                # C++ wrapper for FFI rules
                cat > bin/cxx << 'CXX_WRAPPER_EOF'
              #!/usr/bin/env bash
              set -euo pipefail
              get_config() { grep "^''${1} = " .buckconfig.local 2>/dev/null | cut -d'=' -f2 | tr -d ' '; }
              CXX=$(get_config "cxx")
              INCLUDE_ARGS=(
                  "-resource-dir" "$(get_config clang_resource_dir)"
                  "-isystem" "$(get_config gcc_include)"
                  "-isystem" "$(get_config gcc_include_arch)"
                  "-isystem" "$(get_config glibc_include)"
              )
              exec "$CXX" "''${INCLUDE_ARGS[@]}" "$@"
              CXX_WRAPPER_EOF
                chmod +x bin/cxx

                # compile_commands.json generator for clangd/clang-tidy
                cat > bin/compdb << 'COMPDB_EOF'
              #!/usr/bin/env bash
              # Generate compile_commands.json for clangd/clang-tidy
              # Usage: compdb [targets...]
              # If no targets, generates for all C++ targets in the project
              set -euo pipefail

              TARGETS="''${@:-//...}"

              echo "Generating compile_commands.json for: $TARGETS"
              COMPDB_PATH=$(buck2 bxl prelude//cxx/tools/compilation_database.bxl:generate -- --targets $TARGETS 2>/dev/null | tail -1)

              if [ -n "$COMPDB_PATH" ] && [ -f "$COMPDB_PATH" ]; then
                cp "$COMPDB_PATH" compile_commands.json
                echo "Generated compile_commands.json ($(jq length compile_commands.json) entries)"
              else
                echo "Failed to generate compile_commands.json" >&2
                exit 1
              fi
              COMPDB_EOF
                chmod +x bin/compdb
            ''}

            echo "Generated bin/ wrappers for Buck2 toolchains"
          ''
          +
            lib.optionalString
              (isLinux && cfg.toolchain.cxx.enable && cfg.compdb.enable && cfg.compdb.auto-generate)
              ''
                # Auto-generate compile_commands.json for clangd
                if command -v buck2 &>/dev/null; then
                  echo "Generating compile_commands.json..."
                  TARGETS="${lib.concatStringsSep " " cfg.compdb.targets}"
                  COMPDB_PATH=$(buck2 bxl prelude//cxx/tools/compilation_database.bxl:generate -- --targets $TARGETS 2>/dev/null | tail -1) || true
                  if [ -n "$COMPDB_PATH" ] && [ -f "$COMPDB_PATH" ]; then
                    cp "$COMPDB_PATH" compile_commands.json
                    echo "Generated compile_commands.json ($(jq length compile_commands.json 2>/dev/null || echo '?') entries)"
                  fi
                fi
              '';
      in
      {
        # Export toolchain configuration for other modules
        straylight.build = {
          inherit buck2-toolchain buckconfig-local;
          shellHook = buildShellHook;

          # Packages for devshell
          packages =
            lib.optionals (isLinux && llvm-git != null && cfg.toolchain.cxx.enable) [ llvm-git ]
            ++ lib.optionals (isLinux && nvidia-sdk != null && cfg.toolchain.nv.enable) [ nvidia-sdk ]
            ++ lib.optionals cfg.toolchain.haskell.enable [ ghcForBuck2 ]
            ++ lib.optionals (cfg.toolchain.rust.enable && pkgs ? rustc) [
              pkgs.rustc
              pkgs.cargo
              pkgs.clippy
              pkgs.rustfmt
            ]
            ++ lib.optionals (cfg.toolchain.lean.enable && pkgs ? lean4) [ pkgs.lean4 ]
            ++ lib.optionals cfg.toolchain.python.enable [ pythonEnv ]
            ++ lib.optionals (pkgs ? buck2) [ pkgs.buck2 ];
        };

      };
  };
}
