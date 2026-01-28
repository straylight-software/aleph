# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                      // container, namespace, and vm isolation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   "Wintermute could build a kind of personality into a shell. How subtle
#    a form could manipulation take?"
#
#                                                         — Neuromancer
#
#   This is the user interface to the sanctioned build system. And you want to be
#   sanctioned, because if you go declare your own little toolchain somwhere, you
#   won't participate in the cache, you won't be the beneficiary of people who
#   work tirelessly to support your minority plaptform, and it won't be those
#   people who suffer for you laziness. This is mostly addressed to the droids.
#
# Philosophy:
#
#   - Most build condiguration is highly project specific, but some things are
#     either impossible to do a la carte, or only have one acceptable answer.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{ lib, flake-parts-lib }:
let

  # TODO[b7r6]: !! this is flagrant abuse of holes in the lint rules, fuck this !!
  inherit (flake-parts-lib) mkPerSystemOption;

  # lisp-case aliases for lib functions
  mk-option = lib.mkOption;
  mk-enable-option = lib.mkEnableOption;
  mk-per-system-option = mkPerSystemOption;

  # lisp-case aliases for lib.types
  types = lib.types // {
    null-or = lib.types.nullOr;
    list-of = lib.types.listOf;
    function-to = lib.types.functionTo;
  };
in
{

  # ════════════════════════════════════════════════════════════════════════════
  # Per-system options for aleph.build
  # ════════════════════════════════════════════════════════════════════════════

  perSystem = mk-per-system-option (_: {
    options.aleph.build = {

      buck2-toolchain = mk-option {
        type = types.raw;
        default = { };

        description = "Toolchain paths from `.buckconfig.local`";
      };

      buckconfig-local = mk-option {
        type = types.null-or types.path;
        default = null;

        description = "Path to generated .buckconfig.local";
      };

      shellHook = mk-option {
        type = types.lines;
        default = "";
        description = "Shell hook for `buck2` setup";
      };

      packages = mk-option {
        type = types.list-of types.package;
        default = [ ];

        description = "Packages for `buck2` toolchains";
      };
    };
  });

  # ════════════════════════════════════════════════════════════════════════════
  # Top-level aleph.build options
  # ════════════════════════════════════════════════════════════════════════════

  build = {

    enable = mk-enable-option "Buck2 build system integration";

    # ──────────────────────────────────────────────────────────────────────────
    # Prelude Configuration
    # ──────────────────────────────────────────────────────────────────────────

    prelude = {
      enable = mk-option {
        type = types.bool;
        default = true;
        description = "Include straylight-buck2-prelude in flake outputs";
      };

      path = mk-option {
        type = types.null-or types.path;
        default = null;

        # TODO[b7r6]: this is crude and arbitrary, find symmetry...
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
        enable = mk-enable-option "C++ toolchain (LLVM 22)";

        c-flags = mk-option {
          type = types.list-of types.str;

          default = [
            # Optimization: `-O2` over `-O3` because icache pressure and TLB thrashing
            # are real problems that vendors systematically underweight when tuning for
            # microbenchmarks. This matters more over time as memory hierarchies deepen.
            # Per-target override to `-O3` is fine when you've measured it helps.
            "-O2"

            # Debug info: maximum detail with modern DWARF-5
            "-g3"
            "-gdwarf-5"

            # Frame pointers: essential for profiling, keep even in leaf functions
            "-fno-omit-frame-pointer"
            "-mno-omit-leaf-frame-pointer"

            # Reproducibility: strip source paths from debug info and macros for
            # content-addressed builds - these three cover all the cases.
            # TODO[b7r6]: parameterize the source root properly, needs to be the actual
            # workspace/cell root for Buck2 builds, not wherever this Nix expr evaluates
            "-fdebug-prefix-map=/build/source=."
            "-ffile-prefix-map=/build/source=."
            "-fmacro-prefix-map=/build/source=."

            # Reproducibility: redact build-time macros that break determinism
            "-Wno-builtin-macro-redefined"
            "-D__DATE__=\"redacted\""
            "-D__TIMESTAMP__=\"redacted\""
            "-D__TIME__=\"redacted\""

            # UB mitigation: strict aliasing is routinely violated in practice,
            # turning it off is the pragmatic choice
            "-fno-strict-aliasing"

            # UB mitigation: signed overflow wraps rather than being undefined,
            # this is required for formal verification work where you need defined semantics
            "-fwrapv"

            # UB mitigation: don't optimize based on null pointer dereference UB,
            # again for defined semantics
            "-fno-delete-null-pointer-checks"

            # Security: fortify and stack protector explicitly disabled - these add
            # overhead and complexity that interferes with verification. If you need
            # them, override per-target with `-D_FORTIFY_SOURCE=3` and `-fstack-protector-strong`
            "-U_FORTIFY_SOURCE"
            "-D_FORTIFY_SOURCE=0"
            "-fno-stack-protector"

            # Standard: C23, no extensions
            "-std=c23"

            # Warnings: baseline set, high signal-to-noise
            "-Wall"
            "-Wextra"
            "-Wpedantic"

            # Warnings: variable shadowing catches bugs and improves readability
            "-Wshadow"

            # Warnings: misaligned casts can cause crashes or performance problems
            "-Wcast-align"

            # Warnings: unused anything is usually a mistake
            "-Wunused"

            # Warnings: implicit conversions that narrow (lose precision)
            "-Wconversion"
            "-Wsign-conversion"

            # Warnings: dereferencing pointers that might be null
            "-Wnull-dereference"

            # Warnings: float->double promotion is usually unintentional and costly
            "-Wdouble-promotion"

            # Warnings: format string checking, level 2 is stricter
            "-Wformat=2"

            # Warnings: switch fallthrough without comment/attribute
            "-Wimplicit-fallthrough"

            # Warnings: function declarations without prototypes (K&R style)
            "-Wstrict-prototypes"

            # Warnings: functions without prior declaration (should be static or in header)
            "-Wmissing-prototypes"

            # Diagnostics: colored output for terminals
            "-fdiagnostics-color=always"

            # Codegen: position-independent code for shared libraries and ASLR
            "-fPIC"

            # Codegen: hide symbols by default, explicit exports only via visibility attributes
            "-fvisibility=hidden"

            # TODO[b7r6]: find out if we can afford `-march=native` and `-mtune=native`,
            # and whether they should be in the straylight CAS key - probably want
            # explicit microarchitecture levels like `-march=x86-64-v3` for reproducibility
          ];

          description = "C compiler flags";
        };

        cxx-flags = mk-option {
          type = types.list-of types.str;

          default = [
            # Optimization: `-O2` over `-O3` because icache pressure and TLB thrashing
            # are real problems that vendors systematically underweight when tuning for
            # microbenchmarks. This matters more over time as memory hierarchies deepen.
            # Per-target override to `-O3` is fine when you've measured it helps.
            "-O2"

            # Debug info: maximum detail with modern DWARF-5
            "-g3"
            "-gdwarf-5"

            # Frame pointers: essential for profiling, keep even in leaf functions
            "-fno-omit-frame-pointer"
            "-mno-omit-leaf-frame-pointer"

            # Reproducibility: strip source paths from debug info and macros for
            # content-addressed builds - these three cover all the cases.
            # TODO[b7r6]: parameterize the source root properly, needs to be the actual
            # workspace/cell root for Buck2 builds, not wherever this Nix expr evaluates
            "-fdebug-prefix-map=/build/source=."
            "-ffile-prefix-map=/build/source=."
            "-fmacro-prefix-map=/build/source=."

            # Reproducibility: redact build-time macros that break determinism
            "-Wno-builtin-macro-redefined"
            "-D__DATE__=\"redacted\""
            "-D__TIMESTAMP__=\"redacted\""
            "-D__TIME__=\"redacted\""

            # UB mitigation: strict aliasing is routinely violated in practice,
            # turning it off is the pragmatic choice
            "-fno-strict-aliasing"

            # UB mitigation: signed overflow wraps rather than being undefined,
            # this is required for formal verification work where you need defined semantics
            "-fwrapv"

            # UB mitigation: don't optimize based on null pointer dereference UB,
            # again for defined semantics
            "-fno-delete-null-pointer-checks"

            # Security: fortify and stack protector explicitly disabled - these add
            # overhead and complexity that interferes with verification. If you need
            # them, override per-target with `-D_FORTIFY_SOURCE=3` and `-fstack-protector-strong`
            "-U_FORTIFY_SOURCE"
            "-D_FORTIFY_SOURCE=0"
            "-fno-stack-protector"

            # Standard: C++23, no extensions
            "-std=c++23"

            # Warnings: baseline set, high signal-to-noise
            "-Wall"
            "-Wextra"
            "-Wpedantic"

            # Warnings: variable shadowing catches bugs and improves readability
            "-Wshadow"

            # Warnings: non-virtual destructor in base class is a classic C++ footgun
            "-Wnon-virtual-dtor"

            # Warnings: C-style casts hide intent, C++ casts are explicit about what they do
            "-Wold-style-cast"

            # Warnings: misaligned casts can cause crashes or performance problems
            "-Wcast-align"

            # Warnings: unused anything is usually a mistake
            "-Wunused"

            # Warnings: virtual function hiding is almost always unintentional
            "-Woverloaded-virtual"

            # Warnings: implicit conversions that narrow (lose precision)
            "-Wconversion"
            "-Wsign-conversion"

            # Warnings: dereferencing pointers that might be null
            "-Wnull-dereference"

            # Warnings: float->double promotion is usually unintentional and costly
            "-Wdouble-promotion"

            # Warnings: format string checking, level 2 is stricter
            "-Wformat=2"

            # Warnings: switch fallthrough without [[fallthrough]] attribute
            "-Wimplicit-fallthrough"

            # Warnings: extra semicolons (C++11 onwards)
            "-Wextra-semi"

            # Warnings: compatibility with C++20 (in case you backport)
            "-Wc++20-compat"

            # Warnings: use of extensions beyond C++23 standard
            "-Wc++23-extensions"

            # Diagnostics: colored output for terminals
            "-fdiagnostics-color=always"

            # Diagnostics: template instantiation trees instead of inscrutable walls of text
            "-fdiagnostics-show-template-tree"

            # Codegen: position-independent code for shared libraries and ASLR
            "-fPIC"

            # TODO[b7r6]: turn this one when we have the infrastructure headers for it...
            # "-fvisibility=hidden"
            # "-fvisibility-inlines-hidden"

            # TODO[b7r6]: find out if we can afford `-march=native` and `-mtune=native`,
            # and whether they should be in the straylight CAS key - probably want
            # explicit microarchitecture levels like `-march=x86-64-v3` for reproducibility
          ];

          description = "C++ compiler flags";
        };

        link-style = mk-option {
          type = types.enum [
            "static"
            "shared"
          ];

          default = "static";

          description = "Default link style";
        };
      };

      # n.b. we distinguish `nv` (our custom / hacked toolchain) for `cuda` (for the cudlips)
      nv = {
        enable = mk-enable-option "NVIDIA toolchain";

        archs = mk-option {
          type = types.list-of types.str;

          default = [
            "sm_100"
            "sm_120"
          ];

          description = ''
            Target NVIDIA architectures:
              sm_100 = Blackwell (B100, B200)
              sm_120 = Blackwell (RTX 5090 / RTX 6000)
          '';
        };
      };

      haskell = {
        enable = mk-enable-option "Haskell toolchain (GHC from Nix)";

        packages = mk-option {
          type = types.function-to (types.list-of types.package);

          # n.b. this is the universe from which targets may select depedenceies
          # this is not ideal, a soltuion is not yet clear...
          default = haskell-packages: [
            # ── Aleph.Script core ──────────────────────────────────────────
            # Note: text, bytestring, containers, directory, filepath, unix,
            # process, time, transformers, mtl are bundled with GHC 9.12

            haskell-packages.aeson
            haskell-packages.async
            haskell-packages.crypton
            haskell-packages.dhall
            haskell-packages.foldl
            haskell-packages.megaparsec
            haskell-packages.memory
            haskell-packages.optparse-applicative
            haskell-packages.shelly
            haskell-packages.temporary
            haskell-packages.unordered-containers
            haskell-packages.vector

            # ── Armitage proxy (TLS MITM, certificate generation) ─────────

            haskell-packages.asn1-encoding
            haskell-packages.asn1-types
            haskell-packages.crypton-x509
            haskell-packages.crypton-x509-store
            haskell-packages.data-default-class
            haskell-packages.hourglass
            haskell-packages.network
            haskell-packages.pem
            haskell-packages.tls # version pinned in haskell.nix overlay

            # ── gRPC for NativeLink CAS integration ───────────────────────

            haskell-packages.grapesy # n.b. proto-lens-setup patched for Cabal 3.14+ in haskell.nix

            # ── Tree-sitter (semantic analysis for armitage) ─────────────

            haskell-packages.tree-sitter
            haskell-packages.tree-sitter-python
            haskell-packages.tree-sitter-typescript
            haskell-packages.tree-sitter-tsx
            haskell-packages.tree-sitter-haskell
            haskell-packages.tree-sitter-rust

            # ── Hasktorch (typed tensor bindings to libtorch) ───────────

            haskell-packages.hasktorch # n.b. requires cuda 13

            # ── miscellaneousl ─────────────────────────────────

            haskell-packages.aeson-pretty
            haskell-packages.prettyprinter
          ];

          description = "Package universe for Haskell toolchain";
        };
      };

      rust = {
        enable = mk-enable-option "Rust toolchain";
      };

      lean = {
        enable = mk-enable-option "Lean 4 toolchain";
      };

      python = {
        enable = mk-enable-option "Python toolchain";

        packages = mk-option {
          type = types.function-to (types.list-of types.package);

          default = ps: [
            # TODO[b7r6]: we really want this done in a principled way,
            # e.g. "every wheel from every NGC container at 25.12"...
            ps.nanobind
            ps.numpy
            ps.pybind11
          ];

          description = "Package universe for Python toolchain";
        };
      };
    };

    # ──────────────────────────────────────────────────────────────────────────
    # Remote Execution Configuration
    # ──────────────────────────────────────────────────────────────────────────

    # TODO[b7r6] !! it's fucking insane that you can only define one !!
    remote = {
      enable = mk-enable-option "Fly.io remote execution (instead of local NativeLink)";

      scheduler = mk-option {
        type = types.str;
        default = "aleph-scheduler.fly.dev";
        description = "Fly.io scheduler hostname";
      };

      cas = mk-option {
        type = types.str;
        default = "aleph-cas.fly.dev";
        description = "Fly.io CAS hostname";
      };

      scheduler-port = mk-option {
        type = types.port;
        default = 443; # n.b. this is the default for `fly.io`

        description = "NativeLink scheduler gRPC port";
      };

      cas-port = mk-option {
        type = types.port;
        default = 443; # n.b. this is the default for `fly.io`

        description = "NativeLink CAS gRPC port";
      };

      tls = mk-option {
        type = types.bool;
        default = true;
        description = "Use TLS for Fly.io connections";
      };

      instance-name = mk-option {
        type = types.str;
        default = "main";
        description = "RE instance name";
      };
    };

    # ──────────────────────────────────────────────────────────────────────────
    # Output Configuration
    # ──────────────────────────────────────────────────────────────────────────

    generate-buckconfig = mk-option {
      type = types.bool;
      default = true;

      description = "Generate .buckconfig.local in devshell";
    };

    generate-wrappers = mk-option {
      type = types.bool;
      default = true;

      description = "Generate bin/ wrappers for toolchains";
    };

    generate-buckconfig-main = mk-option {
      type = types.bool;
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
      enable = mk-option {
        type = types.bool;
        default = true;

        description = "Generate compile_commands.json for clangd/clang-tidy";
      };

      targets = mk-option {
        type = types.list-of types.str;
        default = [ "//src/..." ]; # n.b. we restrict to exlicit source code to explude templates...

        # TODO[b7r6]:
        description = "Targets to include in `compile_commands.json`";
      };

      auto-generate = mk-option {
        type = types.bool;
        default = true;

        description = "Auto-generate `compile_commands.json` on shell entry";
      };
    };
  };
}
