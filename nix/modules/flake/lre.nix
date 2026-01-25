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
#   aleph.lre.enable = true;
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

  # Lisp-case aliases for lib.* functions
  map-attrs' = lib.mapAttrs';
  name-value-pair = lib.nameValuePair;
  to-upper = lib.toUpper;
  replace-strings = builtins.replaceStrings;
  optional-string = lib.optionalString;
  optional = lib.optional;

  cfg = config.aleph.lre;
  # Remote execution config (from build module, with defaults if not present)
  remote-cfg =
    config.aleph.build.remote or {
      enable = false;
      scheduler = "aleph-scheduler.fly.dev";
      cas = "aleph-cas.fly.dev";
      scheduler-port = 50051;
      cas-port = 50052;
      tls = true;
      instance-name = "main";
    };
in
{
  _class = "flake";

  # ════════════════════════════════════════════════════════════════════════════
  # Per-system options for aleph.lre
  # ════════════════════════════════════════════════════════════════════════════
  options.perSystem = mkPerSystemOption (
    { lib, ... }:
    {
      options.aleph.lre = {
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

  options.aleph.lre = {
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

        # Render Dhall template with environment variables
        render-dhall =
          name: src: vars:
          let
            env-vars = map-attrs' (
              k: v: name-value-pair (to-upper (replace-strings [ "-" ] [ "_" ] k)) (toString v)
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

        # Get nativelink from inputs or nixpkgs
        nativelink =
          if inputs ? nativelink then
            inputs.nativelink.packages.${system}.default or inputs.nativelink.packages.${system}.nativelink
              or null
          else
            null;

        # Buck2 RE configuration file (generated from Dhall template)
        # Use remote config if build.remote.enable is true, otherwise local
        buckconfig-re-file =
          if remote-cfg.enable then
            render-dhall "lre-buckconfig-remote.ini" ./scripts/lre-buckconfig-remote.dhall {
              scheduler = remote-cfg.scheduler;
              scheduler_port = toString remote-cfg.scheduler-port;
              cas = remote-cfg.cas;
              cas_port = toString remote-cfg.cas-port;
              tls = if remote-cfg.tls then "true" else "false";
              protocol = if remote-cfg.tls then "grpcs" else "grpc";
              instance_name = remote-cfg.instance-name;
            }
          else
            render-dhall "lre-buckconfig.ini" ./scripts/lre-buckconfig.dhall {
              port = toString cfg.port;
              instance_name = cfg.instance-name;
            };

        # lre-start script (generated from Dhall template)
        lre-start =
          let
            nativelink-bin = if nativelink != null then "${nativelink}/bin/nativelink" else "nativelink";
            script-drv = render-dhall "lre-start" ./scripts/lre-start.dhall {
              default_workers = toString cfg.workers;
              default_port = toString cfg.port;
              nativelink = nativelink-bin;
            };
          in
          pkgs.stdenv.mkDerivation {
            name = "lre-start";
            src = script-drv;
            dontUnpack = true;
            installPhase = ''
              mkdir -p $out/bin
              cp $src $out/bin/lre-start
              chmod +x $out/bin/lre-start
            '';

            meta = {
              description = "Start NativeLink for local remote execution with Buck2";
            };
          };

        # Shell hook to append RE config to .buckconfig.local
        lre-shell-hook =
          let
            mode-msg =
              if remote-cfg.enable then
                "Fly.io remote (${remote-cfg.scheduler})"
              else
                "local NativeLink (port ${toString cfg.port})";
          in
          optional-string isLinux ''
            # Append RE configuration to .buckconfig.local
            # (build.nix shellHook creates .buckconfig.local, we append to it)
            if [ -f ".buckconfig.local" ]; then
              # Check if RE config already present
              if ! grep -q "buck2_re_client" .buckconfig.local 2>/dev/null; then
                cat ${buckconfig-re-file} >> .buckconfig.local
                echo "Appended RE config to .buckconfig.local (${mode-msg})"
              fi
            else
              echo "Warning: .buckconfig.local not found, RE config not added"
            fi
          '';

      in
      {
        aleph.lre = {
          shellHook = lre-shell-hook;
          packages = [ lre-start ] ++ optional (nativelink != null) nativelink;
          # Expose individual packages for flake outputs
          inherit lre-start nativelink;
        };

        # Expose lre-start as a runnable package
        packages.lre-start = lre-start;
      };
  };
}
