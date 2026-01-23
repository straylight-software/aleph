# nix/modules/flake/lre.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // lre //
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
# Local Remote Execution (LRE) via NativeLink.
#
# This module provides:
#   - NativeLink binary for local CAS/scheduler
#   - Buck2 RE configuration for .buckconfig.local
#   - lre-start script for easy startup
#   - Shell hook to append RE config
#
# USAGE:
#
#   imports = [ aleph.modules.flake.lre ];
#
#   aleph-naught.lre.enable = true;
#
# Then in your devshell:
#   lre-start --workers=4
#   buck2 build --prefer-remote //...
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
  cfg = config.aleph-naught.lre;
in
{
  _class = "flake";

  # ════════════════════════════════════════════════════════════════════════════
  # Per-system options for straylight.lre
  # ════════════════════════════════════════════════════════════════════════════
  options.perSystem = mkPerSystemOption (
    { lib, ... }:
    {
      options.straylight.lre = {
        shellHook = lib.mkOption {
          type = lib.types.lines;
          default = "";
          description = "Shell hook for LRE setup";
        };
        packages = lib.mkOption {
          type = lib.types.listOf lib.types.package;
          default = [ ];
          description = "LRE packages (nativelink, lre-start)";
        };
        lre-start = lib.mkOption {
          type = lib.types.nullOr lib.types.package;
          default = null;
          description = "The lre-start package";
        };
        nativelink = lib.mkOption {
          type = lib.types.nullOr lib.types.package;
          default = null;
          description = "The nativelink package";
        };
      };
    }
  );

  options.aleph-naught.lre = {
    enable = lib.mkEnableOption "NativeLink Local Remote Execution";

    port = lib.mkOption {
      type = lib.types.port;
      default = 50051;
      description = "Port for NativeLink CAS/scheduler";
    };

    workers = lib.mkOption {
      type = lib.types.int;
      default = 4;
      description = "Default number of worker processes";
    };

    instance-name = lib.mkOption {
      type = lib.types.str;
      default = "main";
      description = "RE instance name";
    };
  };

  # ════════════════════════════════════════════════════════════════════════════
  # Implementation
  # ════════════════════════════════════════════════════════════════════════════
  config = lib.mkIf cfg.enable {
    perSystem =
      {
        pkgs,
        system,
        lib,
        ...
      }:
      let
        inherit (pkgs.stdenv) isLinux;

        # Get nativelink from inputs or nixpkgs
        nativelink =
          if inputs ? nativelink then
            inputs.nativelink.packages.${system}.default or inputs.nativelink.packages.${system}.nativelink
              or null
          else
            null;

        # Buck2 RE configuration file (generated from template)
        buckconfig-re-file = pkgs.substitute {
          src = ./scripts/lre-buckconfig.ini;
          substitutions = [
            "--replace-fail"
            "@port@"
            (toString cfg.port)
            "--replace-fail"
            "@instanceName@"
            cfg.instance-name
          ];
        };

        # lre-start script (external file with replaceVars)
        lre-start =
          let
            nativelinkBin = if nativelink != null then "${nativelink}/bin/nativelink" else "nativelink"; # Fallback to PATH
          in
          pkgs.stdenv.mkDerivation {
            name = "lre-start";
            src = ./scripts/lre-start.bash;
            dontUnpack = true;
            installPhase = ''
              mkdir -p $out/bin
              substitute $src $out/bin/lre-start \
                --replace-fail "@defaultWorkers@" "${toString cfg.workers}" \
                --replace-fail "@defaultPort@" "${toString cfg.port}" \
                --replace-fail "@nativelink@" "${nativelinkBin}"
              chmod +x $out/bin/lre-start
            '';

            meta = {
              description = "Start NativeLink for local remote execution with Buck2";
            };
          };

        # Shell hook to append RE config to .buckconfig.local
        lreShellHook = lib.optionalString isLinux ''
          # Append RE configuration to .buckconfig.local
          # (build.nix shellHook creates .buckconfig.local, we append to it)
          if [ -f ".buckconfig.local" ]; then
            # Check if RE config already present
            if ! grep -q "buck2_re_client" .buckconfig.local 2>/dev/null; then
              cat ${buckconfig-re-file} >> .buckconfig.local
              echo "Appended NativeLink RE config to .buckconfig.local"
            fi
          else
            echo "Warning: .buckconfig.local not found, LRE config not added"
          fi
        '';

      in
      {
        straylight.lre = {
          shellHook = lreShellHook;
          packages = [ lre-start ] ++ lib.optional (nativelink != null) nativelink;
          # Expose individual packages for flake outputs
          inherit lre-start nativelink;
        };

        # Expose lre-start as a runnable package
        packages.lre-start = lre-start;
      };
  };
}
