# nix/modules/flake/build/options.nix
#
# Options for aleph-naught.build module
#
{ lib, flake-parts-lib }:
let
  inherit (flake-parts-lib) mkPerSystemOption;
in
{
  # ════════════════════════════════════════════════════════════════════════════
  # Per-system options for straylight.build
  # ════════════════════════════════════════════════════════════════════════════
  perSystem = mkPerSystemOption (
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

  # ════════════════════════════════════════════════════════════════════════════
  # Top-level aleph-naught.build options
  # ════════════════════════════════════════════════════════════════════════════
  build = {
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
          # Full Aleph.Script dependencies - matches src/tools/scripts/BUCK SCRIPT_PACKAGES
          # and nix/overlays/script.nix hsDeps
          default =
            hp: with hp; [
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
        enable = lib.mkEnableOption "Python toolchain (with nanobind/pybind11)";

        packages = lib.mkOption {
          type = lib.types.functionTo (lib.types.listOf lib.types.package);
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
      enable = lib.mkEnableOption "Fly.io remote execution (instead of local NativeLink)";

      scheduler = lib.mkOption {
        type = lib.types.str;
        default = "aleph-scheduler.fly.dev";
        description = "Fly.io scheduler hostname";
      };

      cas = lib.mkOption {
        type = lib.types.str;
        default = "aleph-cas.fly.dev";
        description = "Fly.io CAS hostname";
      };

      scheduler-port = lib.mkOption {
        type = lib.types.port;
        default = 50051;
        description = "Scheduler gRPC port";
      };

      cas-port = lib.mkOption {
        type = lib.types.port;
        default = 50052;
        description = "CAS gRPC port";
      };

      tls = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Use TLS for Fly.io connections";
      };

      instance-name = lib.mkOption {
        type = lib.types.str;
        default = "main";
        description = "RE instance name";
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
}
