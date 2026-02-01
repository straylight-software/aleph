# nix/modules/flake/build/options.nix
#
# Options for aleph.build module
#
{ lib, flake-parts-lib }:
let
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
        description = "Buck2 toolchain paths from .buckconfig.local";
      };
      buckconfig-local = mk-option {
        type = types.null-or types.path;
        default = null;
        description = "Path to generated .buckconfig.local";
      };
      shellHook = mk-option {
        type = types.lines;
        default = "";
        description = "Shell hook for Buck2 setup";
      };
      packages = mk-option {
        type = types.list-of types.package;
        default = [ ];
        description = "Packages for Buck2 toolchains";
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

        cxx-flags = mk-option {
          type = types.list-of types.str;
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

        link-style = mk-option {
          type = types.enum [
            "static"
            "shared"
          ];
          default = "static";
          description = "Default link style";
        };
      };

      nv = {
        enable = mk-enable-option "NVIDIA toolchain (clang + nvidia-sdk)";

        archs = mk-option {
          type = types.list-of types.str;
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
        enable = mk-enable-option "Haskell toolchain (GHC from Nix)";

        packages = mk-option {
          type = types.function-to (types.list-of types.package);
          # Full Aleph.Script dependencies - matches src/tools/scripts/BUCK SCRIPT_PACKAGES
          # and nix/overlays/script.nix hsDeps
          #
          # Also includes armitage proxy dependencies (TLS MITM, certificates)
          # and grapesy for future NativeLink CAS integration.
          default =
            hp: with hp; [
              # ── Aleph.Script core ──────────────────────────────────────────
              megaparsec
              text
              shelly
              foldl
              aeson
              dhall
              directory
              filepath
              crypton
              memory
              unordered-containers
              vector
              unix
              async
              bytestring
              process
              containers
              transformers
              mtl
              time
              optparse-applicative

              # ── Armitage proxy (TLS MITM, certificate generation) ─────────
              network
              tls # version pinned in haskell.nix overlay
              crypton-x509
              crypton-x509-store
              data-default-class
              pem
              asn1-types
              asn1-encoding
              hourglass

              # ── gRPC for NativeLink CAS integration ───────────────────────
              # proto-lens-setup patched for Cabal 3.14+ in haskell.nix
              grapesy
            ];
          description = "Haskell packages for Buck2 toolchain (receives haskellPackages)";
        };
      };

      rust = {
        enable = mk-enable-option "Rust toolchain";
      };

      lean = {
        enable = mk-enable-option "Lean 4 toolchain";
      };

      python = {
        enable = mk-enable-option "Python toolchain (with nanobind/pybind11)";

        packages = mk-option {
          type = types.function-to (types.list-of types.package);
          default = ps: [
            ps.nanobind
            ps.pybind11
            ps.numpy
          ];
          description = "Python packages for Buck2 toolchain";
        };
      };
    };

    # ──────────────────────────────────────────────────────────────────────────
    # Remote Execution Configuration
    # ──────────────────────────────────────────────────────────────────────────
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
        default = 50051;
        description = "Scheduler gRPC port";
      };

      cas-port = mk-option {
        type = types.port;
        default = 50052;
        description = "CAS gRPC port";
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
        default = [ "//..." ];
        description = "Buck2 targets to include in compile_commands.json";
      };

      auto-generate = mk-option {
        type = types.bool;
        default = false;
        description = ''
          Auto-generate compile_commands.json on shell entry.
          Can be slow for large projects. Use bin/compdb manually instead.
        '';
      };
    };
  };
}
