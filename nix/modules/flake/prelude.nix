# nix/modules/flake/prelude.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                           // the aleph prelude //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "The Sprawl was a long strange way home over the Pacific now, and he
#      was no console man, no cyberspace cowboy."
#
#                                                         — Neuromancer
#
# A flake-parts module that exposes the Aleph Prelude from the overlay.
#
# The prelude overlay (nix/overlays/prelude.nix) provides:
#   - aleph.prelude: functional library
#   - aleph.stdenv: build environment matrix
#   - aleph.cross: cross-compilation targets
#   - aleph.platform: platform detection
#   - aleph.gpu: GPU architecture metadata
#   - aleph.turing-registry: non-negotiable build flags
#
# Access via _module.args:
#   perSystem = { prelude, pkgs, ... }: { ... }
#
# Or via config:
#   perSystem = { config, ... }: let p = config.aleph.prelude; in { ... }
#
# See RFC-003: docs/languages/nix/rfc/003-prelude.md
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
{
  lib,
  flake-parts-lib,
  ...
}:
let
  inherit (flake-parts-lib) mkPerSystemOption;
in
{
  _class = "flake";

  # ────────────────────────────────────────────────────────────────────────────
  # // options //
  # ────────────────────────────────────────────────────────────────────────────

  options = {
    perSystem = mkPerSystemOption (
      { lib, ... }:
      {
        options.aleph = {
          prelude = lib.mkOption {
            type = lib.types.raw;
            description = "The instantiated Aleph Prelude for this system";
          };
        };
      }
    );
  };

  # ────────────────────────────────────────────────────────────────────────────
  # // config //
  # ────────────────────────────────────────────────────────────────────────────

  config = {
    perSystem =
      { pkgs, system, ... }:
      let
        # Get the prelude from the overlay (must have aleph overlay applied)
        aleph =
          pkgs.aleph or (throw ''
            aleph.prelude requires the aleph overlay to be applied.
            Make sure you import aleph.flakeModules.std or apply the overlay manually.
          '');

        # ──────────────────────────────────────────────────────────────────────
        # // wasm infrastructure //
        # ──────────────────────────────────────────────────────────────────────

        wasm-infra = import ../../prelude/wasm-plugin.nix {
          inherit lib;
          inherit (pkgs) stdenv runCommand writeText;
          ghc-wasm-meta = inputs.ghc-wasm-meta or null;
        };

        # GHC WASM toolchain (if available)
        ghc-wasm =
          if inputs ? ghc-wasm-meta then inputs.ghc-wasm-meta.packages.${system}.all_9_12 else null;

        # ──────────────────────────────────────────────────────────────────────
        # // language toolchains //
        # ──────────────────────────────────────────────────────────────────────

        # Haskell package set - using GHC 9.12 from nixpkgs (stable, well-tested)
        # GHC 9.12 is the latest stable before 9.14's doctest/HLS breakage.
        # This replaces the Mercury GHC approach which had package.cache.lock bugs.
        hs-pkgs = pkgs.haskell.packages.ghc912;

        python =
          let
            version = aleph.versions.python;
            pkg = pkgs.python312;
            pkgs' = pkgs.python312Packages;
          in
          {
            inherit version pkg pkgs';
            interpreter = pkg;
            build = attrs: pkgs'.buildPythonPackage (aleph.translate-attrs attrs);
            app = attrs: pkgs'.buildPythonApplication (aleph.translate-attrs attrs);
            lib = attrs: pkgs'.buildPythonPackage (aleph.translate-attrs attrs // { format = "setuptools"; });
          };

        ghc =
          let
            version = aleph.versions.ghc;
            pkg = hs-pkgs.ghc;
            pkgs' = hs-pkgs;
            build = attrs: pkgs'.mkDerivation (aleph.translate-attrs attrs);
          in
          {
            inherit
              version
              pkg
              pkgs'
              build
              ;
            compiler = pkg;
            app = build;
            lib = build;

            # Turtle shell scripts - compiled Haskell with bash-like ergonomics
            # Benefits over bash:
            #   - Type-safe path/text manipulation
            #   - Proper error handling with ExceptT/Either
            #   - Identical startup time to bash (~2ms compiled vs ~160ms interpreted)
            #   - No shellcheck needed - types catch more
            #
            # Usage:
            #   ghc.turtle-script {
            #     name = "my-script";
            #     src = ./my-script.hs;
            #     deps = [ pkgs.crane pkgs.bwrap ];  # Runtime deps
            #     hs-deps = p: [ p.aeson p.optparse-applicative ];  # Haskell deps
            #   }
            turtle-script =
              {
                name,
                src,
                deps ? [ ],
                hs-deps ? _: [ ],
              }:
              let
                # Base turtle dependencies (always included)
                base-deps =
                  p: with p; [
                    turtle # The Haskell package, not ghc.turtle-script
                    text
                    bytestring
                    foldl
                    unix
                  ];

                # Combined Haskell dependencies
                all-hs-deps = p: base-deps p ++ hs-deps p;

                # GHC with turtle and user's Haskell deps (using ghc912 from nixpkgs)
                ghc-with-deps = hs-pkgs.ghcWithPackages all-hs-deps;
              in
              pkgs.stdenv.mkDerivation {
                inherit name src;
                dontUnpack = true;

                nativeBuildInputs = [ ghc-with-deps ] ++ pkgs.lib.optional (deps != [ ]) pkgs.makeWrapper;
                buildInputs = deps;

                buildPhase = ''
                  runHook preBuild
                  ghc -O2 -Wall -o ${name} $src
                  runHook postBuild
                '';

                installPhase = ''
                  runHook preInstall
                  mkdir -p $out/bin
                  cp ${name} $out/bin/
                  runHook postInstall
                '';

                # Wrap with runtime deps
                postFixup = pkgs.lib.optionalString (deps != [ ]) ''
                  wrapProgram $out/bin/${name} \
                    --prefix PATH : ${pkgs.lib.makeBinPath deps}
                '';

                meta = {
                  description = "Compiled Turtle shell script with type-safe path handling";
                };
              };
          };

        lean =
          let
            version = aleph.versions.lean;
            pkg = pkgs.lean4;
            build =
              attrs:
              aleph.stdenv.default (
                attrs
                // {
                  native-deps = (attrs.native-deps or [ ]) ++ [ pkg ];
                }
              );
          in
          {
            inherit version pkg build;
            lib = build;
          };

        rust =
          let
            version = aleph.versions.rust;
            pkg = pkgs.rustc;
            toolchain = pkgs.rustPlatform;
            crates = pkgs.rustPackages;
            build = attrs: pkgs.rustPlatform.buildRustPackage (aleph.translate-attrs attrs);
          in
          {
            inherit
              version
              pkg
              toolchain
              crates
              build
              ;
            bin = build;
            lib = build;
            staticlib = build;
          };

        cpp = {
          bin = aleph.stdenv.default;
          lib = aleph.stdenv.default;
          staticlib = aleph.stdenv.static;
          header-only = aleph.stdenv.default;
          nvidia = {
            build =
              attrs: (aleph.stdenv.nvidia or aleph.stdenv.default) (builtins.removeAttrs attrs [ "target-gpu" ]);
            kernel =
              attrs: (aleph.stdenv.nvidia or aleph.stdenv.default) (builtins.removeAttrs attrs [ "target-gpu" ]);
            host =
              attrs: (aleph.stdenv.nvidia or aleph.stdenv.default) (builtins.removeAttrs attrs [ "target-gpu" ]);
          };
        };

        # ──────────────────────────────────────────────────────────────────────
        # // fetch //
        # ──────────────────────────────────────────────────────────────────────

        fetch = {
          github = pkgs.fetchFromGitHub;
          gitlab = pkgs.fetchFromGitLab;
          git = pkgs.fetchgit;
          url = pkgs.fetchurl;
          tarball = builtins.fetchTarball;
          fod =
            {
              name,
              hash,
              script,
            }:
            pkgs.runCommand name {
              outputHashMode = "recursive";
              outputHashAlgo = "sha256";
              outputHash = hash;
              nativeBuildInputs = [
                pkgs.curl
                pkgs.jq
              ];
              SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            } script;
        };

        # ──────────────────────────────────────────────────────────────────────
        # // render //
        # ──────────────────────────────────────────────────────────────────────

        render = {
          # JSON: use builtin serialization
          json =
            name: value:
            pkgs.writeTextFile {
              inherit name;
              text = builtins.toJSON value;
            };
          # TOML/YAML/INI: pkgs.formats.*.generate already returns a derivation
          toml = name: value: (pkgs.formats.toml { }).generate name value;
          yaml = name: value: (pkgs.formats.yaml { }).generate name value;
          ini = name: value: (pkgs.formats.ini { }).generate name value;
          # ENV: simple key=value format
          env =
            name: attrs:
            pkgs.writeTextFile {
              inherit name;
              text = lib.concatStringsSep "\n" (lib.mapAttrsToList (k: v: "${k}=${v}") attrs);
            };

          # ────────────────────────────────────────────────────────────────────
          # // dhall renderers //
          # ────────────────────────────────────────────────────────────────────
          #
          # Type-safe templates via Dhall. Replaces @var@ substituteAll patterns.
          #
          # Dhall expressions can read environment variables as typed values:
          #   let path : Text = env:PATH as Text
          #   let count : Natural = env:COUNT
          #
          # Usage:
          #   render.dhall "config.json" ./config.dhall        # Dhall -> JSON
          #   render.dhall-yaml "config.yaml" ./config.dhall   # Dhall -> YAML
          #   render.dhall-text "script.sh" ./script.dhall     # Dhall -> Text
          #   render.dhall-with-vars "out.txt" ./template.dhall { path = "/nix/store/..."; }

          # Dhall -> JSON (dhall-to-json)
          dhall =
            name: src:
            pkgs.runCommand name { nativeBuildInputs = [ pkgs.haskellPackages.dhall-json ]; } ''
              dhall-to-json --file ${src} > $out
            '';

          # Dhall -> YAML (dhall-to-yaml)
          dhall-yaml =
            name: src:
            pkgs.runCommand name { nativeBuildInputs = [ pkgs.haskellPackages.dhall-yaml ]; } ''
              dhall-to-yaml --file ${src} > $out
            '';

          # Dhall -> Text (dhall text)
          # The Dhall expression must evaluate to a Text value.
          dhall-text =
            name: src:
            pkgs.runCommand name { nativeBuildInputs = [ pkgs.haskellPackages.dhall ]; } ''
              dhall text --file ${src} > $out
            '';

          # Dhall -> Text with Nix-injected environment variables
          # Variables are passed via environment to the dhall process.
          # The Dhall file reads them with: let foo = env:FOO as Text
          #
          # Example:
          #   render.dhall-with-vars "script.sh" ./script.dhall {
          #     PATH = "${pkgs.coreutils}/bin";
          #     VERSION = "1.0";
          #   }
          dhall-with-vars =
            name: src: vars:
            let
              # Convert vars attrset to env var exports
              env-vars = lib.mapAttrs' (
                k: v: lib.nameValuePair (lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] k)) (toString v)
              ) vars;
            in
            pkgs.runCommand name
              (
                {
                  nativeBuildInputs = [ pkgs.haskellPackages.dhall ];
                }
                // env-vars
              )
              ''
                dhall text --file ${src} > $out
              '';
        };

        # ──────────────────────────────────────────────────────────────────────
        # // format converters (inline) //
        # ──────────────────────────────────────────────────────────────────────
        #
        # These return strings, not derivations. For use in aleph-script files.

        to-json = builtins.toJSON;
        to-ini =
          attrs:
          lib.concatStringsSep "\n" (
            lib.mapAttrsToList (
              section: values:
              "[${section}]\n"
              + lib.concatStringsSep "\n" (lib.mapAttrsToList (k: v: "${k} = ${toString v}") values)
            ) attrs
          );
        to-env = attrs: lib.concatStringsSep "\n" (lib.mapAttrsToList (k: v: "${k}=${toString v}") attrs);

        # ──────────────────────────────────────────────────────────────────────
        # // aleph-script //
        # ──────────────────────────────────────────────────────────────────────
        #
        # The ONLY sanctioned way to generate files inline. Separates file
        # content (declarative, structured) from shell logic (imperative, minimal).
        #
        # Usage:
        #   configurePhase = aleph-script {
        #     files.".buckconfig.local" = to-ini { cxx = { cc = "${clang}/bin/clang"; }; };
        #     files."config.json" = to-json { key = "value"; };
        #     run = "ln -sf ${prelude} prelude";
        #   };
        #
        # Returns a shell script string suitable for use in derivation phases.

        aleph-script =
          {
            # Attrset of filename -> content (string)
            files ? { },
            # Optional shell commands to run after file generation (keep minimal!)
            run ? "",
          }:
          let
            # Generate file creation commands
            file-cmds = lib.concatStringsSep "\n" (
              lib.mapAttrsToList (
                path: content:
                let
                  # Write content to a temp file in the store, then copy
                  # This avoids heredocs entirely
                  content-file = pkgs.writeText (baseNameOf path) content;
                in
                ''
                  mkdir -p "$(dirname '${path}')"
                  cp ${content-file} '${path}'
                ''
              ) files
            );
          in
          ''
            # Generated by aleph-script
            ${file-cmds}
            ${lib.optionalString (run != "") ''
              # User commands
              ${run}
            ''}
          '';

        # ──────────────────────────────────────────────────────────────────────
        # // write //
        # ──────────────────────────────────────────────────────────────────────

        write = {
          # Plain text file
          text = name: content: pkgs.writeText name content;

          # Shell application with dependencies and shellcheck (THE ONLY WAY)
          # Use this for all shell scripts. No exceptions.
          shell-application =
            {
              name,
              deps ? [ ],
              text,
            }:
            pkgs.writeShellApplication {
              inherit name text;
              runtimeInputs = deps;
            };
        };

        # Top-level alias - the blessed way to write shell scripts
        write-shell-application = write.shell-application;

        # ──────────────────────────────────────────────────────────────────────
        # // mk-package //
        # ──────────────────────────────────────────────────────────────────────
        #
        # No inline code allowed. All build phases must be external files.
        # This ensures:
        # - Scripts are shellcheck'd and resholve'd
        # - Build logic is testable outside Nix
        # - Diffs are meaningful (not buried in Nix strings)

        mk-package =
          {
            pname,
            version,
            src,
            # Scripts directory containing phase scripts
            # Expected files: configure.sh, build.sh, install.sh, check.sh
            scripts ? null,
            # Individual phase scripts (override scripts dir)
            configure-script ? null,
            build-script ? null,
            install-script ? null,
            check-script ? null,
            # Dependencies
            deps ? [ ],
            native-deps ? [ ],
            # Disable phases
            dont-configure ? false,
            dont-build ? false,
            dont-install ? false,
            dont-check ? true,
            # Extra attrs passed through
            ...
          }@args:
          let
            # Resolve script paths
            resolve-script =
              name: explicit: dir:
              if explicit != null then
                explicit
              else if dir != null && builtins.pathExists (dir + "/${name}.sh") then
                dir + "/${name}.sh"
              else
                null;

            configure-src = resolve-script "configure" configure-script scripts;
            build-src = resolve-script "build" build-script scripts;
            install-src = resolve-script "install" install-script scripts;
            check-src = resolve-script "check" check-script scripts;

            # Build phase script wrappers
            mk-phase = script-path: if script-path == null then null else builtins.readFile script-path;

            # Filter out our custom attrs
            extra-attrs = builtins.removeAttrs args [
              "pname"
              "version"
              "src"
              "scripts"
              "configure-script"
              "build-script"
              "install-script"
              "check-script"
              "deps"
              "native-deps"
              "dont-configure"
              "dont-build"
              "dont-install"
              "dont-check"
            ];
          in
          aleph.stdenv.default (
            {
              inherit pname version src;
              buildInputs = deps;
              nativeBuildInputs = native-deps;

              dontConfigure = dont-configure || configure-src == null;
              dontBuild = dont-build || build-src == null;
              dontInstall = dont-install || install-src == null;
              doCheck = !dont-check && check-src != null;
            }
            // lib.optionalAttrs (configure-src != null && !dont-configure) {
              configurePhase = mk-phase configure-src;
            }
            // lib.optionalAttrs (build-src != null && !dont-build) {
              buildPhase = mk-phase build-src;
            }
            // lib.optionalAttrs (install-src != null && !dont-install) {
              installPhase = mk-phase install-src;
            }
            // lib.optionalAttrs (check-src != null && !dont-check) {
              checkPhase = mk-phase check-src;
            }
            // extra-attrs
          );

        # ──────────────────────────────────────────────────────────────────────
        # // script //
        # ──────────────────────────────────────────────────────────────────────

        script = {
          bash =
            {
              name,
              deps ? [ ],
              src,
            }:
            pkgs.writeShellApplication {
              inherit name;
              runtimeInputs = deps;
              text = builtins.readFile src;
            };
          python =
            {
              name,
              deps ? [ ],
              src,
            }:
            pkgs.writers.writePython3Bin name { libraries = deps; } (builtins.readFile src);
          c =
            { name, src }:
            pkgs.runCommandCC name { } ''
              mkdir -p $out/bin
              $CC -O2 -o $out/bin/${name} ${src}
            '';
        };

        # ──────────────────────────────────────────────────────────────────────
        # // opt //
        # ──────────────────────────────────────────────────────────────────────

        opt = {
          enable = desc: lib.mkEnableOption desc;
          str =
            {
              default ? null,
              description ? "",
            }:
            lib.mkOption {
              type = lib.types.str;
              inherit default description;
            };
          int =
            {
              default ? null,
              description ? "",
            }:
            lib.mkOption {
              type = lib.types.int;
              inherit default description;
            };
          bool =
            {
              default ? false,
              description ? "",
            }:
            lib.mkOption {
              type = lib.types.bool;
              inherit default description;
            };
          path =
            {
              default ? null,
              description ? "",
            }:
            lib.mkOption {
              type = lib.types.path;
              inherit default description;
            };
          port =
            {
              default ? null,
              description ? "",
            }:
            lib.mkOption {
              type = lib.types.port;
              inherit default description;
            };
          list-of =
            elemType:
            {
              default ? [ ],
              description ? "",
            }:
            lib.mkOption {
              type = lib.types.listOf elemType;
              inherit default description;
            };
          attrs-of =
            elemType:
            {
              default ? { },
              description ? "",
            }:
            lib.mkOption {
              type = lib.types.attrsOf elemType;
              inherit default description;
            };
          one-of =
            values:
            {
              default ? null,
              description ? "",
            }:
            lib.mkOption {
              type = lib.types.enum values;
              inherit default description;
            };
          package =
            {
              default ? null,
              description ? "",
            }:
            lib.mkOption {
              type = lib.types.package;
              inherit default description;
            };
          submodule =
            options:
            {
              default ? { },
              description ? "",
            }:
            lib.mkOption {
              type = lib.types.submodule { inherit options; };
              inherit default description;
            };
        };

        # ──────────────────────────────────────────────────────────────────────
        # // test //
        # ──────────────────────────────────────────────────────────────────────

        test = {
          # Force evaluation of a list of assertions
          # Returns true if all pass, throws on first failure
          # Usage: test.check [ (test.eq 1 1 "one equals one") (test.eq 2 2 "two equals two") ]
          check =
            assertions:
            builtins.deepSeq (map (
              a:
              assert a.pass;
              a
            ) assertions) true;

          # Create an assertion that checks equality
          eq = actual: expected: msg: {
            pass = actual == expected;
            message = msg;
            inherit actual expected;
          };

          # Create an assertion that checks inequality
          neq = actual: unexpected: msg: {
            pass = actual != unexpected;
            message = msg;
            inherit actual;
            expected = "not ${builtins.toString unexpected}";
          };

          # Create an assertion that checks a predicate
          ok = pred: msg: {
            pass = pred;
            message = msg;
            actual = pred;
            expected = true;
          };

          # Create an assertion that a value is not null
          not-null = value: msg: {
            pass = value != null;
            message = msg;
            actual = value;
            expected = "non-null";
          };

          # Create an assertion that a list contains an element
          contains = elem: list: msg: {
            pass = builtins.elem elem list;
            message = msg;
            actual = list;
            expected = "contains ${builtins.toString elem}";
          };

          # Run a test suite (list of named test groups)
          # Returns attrset of results for integration with nix flake check
          suite =
            name: tests:
            let
              results = map (t: {
                inherit (t) name;
                pass = test.check t.assertions;
              }) tests;
              all-pass = builtins.all (r: r.pass) results;
            in
            {
              inherit name results;
              pass = all-pass;
            };
        };

        # ──────────────────────────────────────────────────────────────────────
        # // typed //
        # ──────────────────────────────────────────────────────────────────────
        #
        # Type-check configuration at eval time, fail fast with clear errors.
        # Based on lib.evalModules like NixOS/home-manager configs.

        typed = {
          # Define a typed configuration schema
          # Returns a function that validates config against the schema
          #
          # Usage:
          #   schema = typed.define {
          #     options.name = lib.mkOption { type = lib.types.str; };
          #     options.count = lib.mkOption { type = lib.types.int; default = 0; };
          #   };
          #   config = schema { name = "foo"; count = 42; };
          #
          define =
            module: config:
            let
              evaluated = lib.evalModules {
                modules = [
                  module
                  { inherit config; }
                ];
              };
            in
            evaluated.config;

          # Common type aliases for convenience
          types = {
            inherit (lib.types)
              str
              int
              bool
              float
              path
              package
              ;
            list = lib.types.listOf;
            attrs = lib.types.attrsOf;
            inherit (lib.types) enum;
            null-or = lib.types.nullOr;
            inherit (lib.types) either;
            one-of = lib.types.oneOf;

            # GPU capability type
            cuda-capability = lib.types.enum [
              "7.0"
              "7.5"
              "8.0"
              "8.6"
              "8.7"
              "8.9"
              "9.0"
              "10.0"
              "12.0"
              "12.1"
            ];

            # GPU architecture type
            cuda-arch = lib.types.enum [
              "volta"
              "turing"
              "ampere"
              "orin"
              "ada"
              "hopper"
              "thor"
              "blackwell"
            ];
          };

          # Helper to create an option with common patterns
          option = {
            required =
              type: description:
              lib.mkOption {
                inherit type description;
              };

            optional =
              type: default: description:
              lib.mkOption {
                inherit type default description;
              };

            enable =
              description:
              lib.mkOption {
                type = lib.types.bool;
                default = false;
                inherit description;
              };
          };
        };

        # ──────────────────────────────────────────────────────────────────────
        # // derivation helpers //
        # ──────────────────────────────────────────────────────────────────────
        #
        # Re-exported from nixpkgs with lisp-case names.
        # These are critical for building proper derivation wrappers.

        drv = {
          # Extend a mkDerivation-like function (preserves finalAttrs, overrideAttrs)
          # See: https://noogle.dev/f/lib/extendMkDerivation
          extend-mk-derivation = lib.extendMkDerivation;

          # Extend a derivation with additional attributes
          extend-derivation = lib.extendDerivation;

          # Make overridable functions
          make-overridable = lib.makeOverridable;

          # Link farm - create derivation with explicit symlinks (declarative)
          link-farm = pkgs.linkFarm;

          # Symlink join - merge packages into one tree
          symlink-join = pkgs.symlinkJoin;

          # Build environment from packages
          build-env = pkgs.buildEnv;

          # Run a command and capture output
          run-command = pkgs.runCommand;
          run-command-local = pkgs.runCommandLocal;

          # DEPRECATED: Use render.dhall-with-vars instead of substitute patterns
          # These are kept temporarily for backward compatibility
          inherit (pkgs) substitute;
          substitute-all = pkgs.substituteAll;
        };

        # ──────────────────────────────────────────────────────────────────────
        # // error context //
        # ──────────────────────────────────────────────────────────────────────
        #
        # Better stack traces via builtins.addErrorContext.
        # Wrap library functions with context for debugging.

        error = {
          # Add context to error messages for better stack traces
          # Usage: error.context "parsing config" (parseConfig cfg)
          context = builtins.addErrorContext;

          # Throw with a message
          inherit (builtins) throw;

          # Abort evaluation (harder failure)
          inherit (builtins) abort;

          # Assert with message - evaluates to x if cond is true, throws otherwise
          # Usage: error.assert-msg (x > 0) "x must be positive" x
          assert-msg =
            cond: msg: x:
            if cond then x else throw msg;

          # Trace for debugging (prints to stderr, returns value)
          inherit (builtins) trace;

          # Trace with label
          trace-val = label: x: builtins.trace "${label}: ${builtins.toJSON x}" x;
        };

        # ──────────────────────────────────────────────────────────────────────
        # // module system //
        # ──────────────────────────────────────────────────────────────────────
        #
        # Type-checked configuration via lib.evalModules.
        # Use for complex configs that benefit from NixOS-style type checking.

        modules = {
          # Evaluate modules (NixOS-style configuration)
          eval = lib.evalModules;

          # Create a submodule type
          inherit (lib.types) submodule;

          # Common option helpers
          mk-option = lib.mkOption;
          mk-enable = lib.mkEnableOption;
          mk-if = lib.mkIf;
          mk-default = lib.mkDefault;
          mk-force = lib.mkForce;
          mk-override = lib.mkOverride;
          mk-merge = lib.mkMerge;

          # Types for options
          inherit (lib) types;
        };

        # ──────────────────────────────────────────────────────────────────────
        # // call-package //
        # ──────────────────────────────────────────────────────────────────────
        #
        # Unified package builder. File extension determines backend:
        #   .hs   → Compile to WASM, evaluate via builtins.wasm
        #   .purs → PureScript WASM (planned)
        #   .nix  → Standard Nix import
        #   .wasm → Pre-compiled WASM, evaluate directly
        #
        # Usage:
        #   nvidia-nccl = call-package ./nvidia-nccl.hs {};
        #   zlib-ng = call-package ./zlib-ng.hs {};
        #

        call-package =
          path: args:
          let
            path-str = toString path;
            ext = lib.last (lib.splitString "." path-str);
            aleph-modules = ../../../src/tools/scripts;

            # Generated Main.hs that wraps the user's Pkg module
            # User files just need: module Pkg where ... pkg = mkDerivation [...]
            wrapper-main = pkgs.writeText "Main.hs" ''
              {-# LANGUAGE ForeignFunctionInterface #-}
              module Main where

              import Aleph.Nix.Value (Value(..))
              import Aleph.Nix.Derivation (drvToNixAttrs)
              import Aleph.Nix (nixWasmInit)
              import qualified Pkg (pkg)

              main :: IO ()
              main = pure ()

              foreign export ccall "nix_wasm_init_v1" initPlugin :: IO ()
              initPlugin :: IO ()
              initPlugin = nixWasmInit

              foreign export ccall "pkg" pkgExport :: Value -> IO Value
              pkgExport :: Value -> IO Value
              pkgExport _args = drvToNixAttrs Pkg.pkg
            '';

            # Build single-file Haskell to WASM
            build-hs-wasm =
              hs-path:
              let
                name = lib.removeSuffix ".hs" (baseNameOf (toString hs-path));
              in
              pkgs.runCommand "${name}.wasm"
                {
                  src = hs-path;
                  nativeBuildInputs = [ ghc-wasm ];
                }
                ''
                  mkdir -p build
                  cd build

                  # Copy Aleph.Nix infrastructure (make writable for .hi files)
                  cp -r ${aleph-modules}/Aleph Aleph
                  chmod -R u+w Aleph

                  # Copy user's package as Pkg.hs, wrapper as Main.hs
                  cp $src Pkg.hs
                  cp ${wrapper-main} Main.hs

                  ${ghc-wasm}/bin/wasm32-wasi-ghc \
                    -optl-mexec-model=reactor \
                    '-optl-Wl,--allow-undefined' \
                    '-optl-Wl,--export=hs_init' \
                    '-optl-Wl,--export=nix_wasm_init_v1' \
                    '-optl-Wl,--export=pkg' \
                    -O2 \
                    Main.hs \
                    -o plugin.wasm

                  ${ghc-wasm}/bin/wasm-opt -O3 plugin.wasm -o $out
                '';
          in
          if ext == "hs" then
            if ghc-wasm == null then
              throw ''
                call-package for .hs files requires ghc-wasm-meta input.
                Add ghc-wasm-meta to your flake inputs or pre-compile to .wasm.
              ''
            else if !(builtins ? wasm) then
              throw ''
                call-package for .hs files requires straylight-nix with builtins.wasm.
                Use straylight-nix or pre-compile to .wasm and use a different evaluator.
              ''
            else
              let
                wasm-drv = build-hs-wasm path;
                # Call "pkg" export which returns the package spec
                spec = builtins.wasm wasm-drv "pkg" args;
              in
              wasm-infra.buildFromSpec { inherit spec pkgs; }

          else if ext == "wasm" then
            if !(builtins ? wasm) then
              throw "call-package for .wasm files requires straylight-nix with builtins.wasm"
            else
              let
                # Assume .wasm files export "pkg" like .hs files
                spec = builtins.wasm path "pkg" args;
              in
              wasm-infra.buildFromSpec { inherit spec pkgs; }

          else if ext == "nix" then
            pkgs.callPackage path args

          else
            throw "call-package: unsupported extension .${ext} (expected .hs, .wasm, or .nix)";

        # ──────────────────────────────────────────────────────────────────────
        # // assembled prelude //
        # ──────────────────────────────────────────────────────────────────────

        prelude = aleph.prelude // {
          # Re-export overlay contents
          inherit (aleph)
            platform
            gpu
            turing-registry
            stdenv
            cross
            versions
            license
            ;

          # pkgs-dependent toolchains
          inherit
            python
            ghc
            lean
            rust
            cpp
            ;

          # pkgs-dependent utilities
          inherit
            fetch
            render
            write
            write-shell-application
            script
            opt
            mk-package
            test
            typed
            call-package
            ;

          # Format converters (return strings for aleph-script)
          inherit to-json to-ini to-env;

          # AlephScript - the ONLY way to generate files inline
          inherit aleph-script;

          # Derivation helpers (lisp-case re-exports)
          inherit drv;

          # Error handling and debugging
          inherit error;

          # Module system helpers
          inherit modules;

          # Raw pkgs access
          inherit pkgs;
        };

      in
      {
        # Expose via config.aleph.prelude
        aleph.prelude = prelude;

        # Expose via _module.args for direct access: { prelude, ... }:
        _module.args.prelude = prelude;
      };
  };
}
