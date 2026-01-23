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

        # Buck2 RE configuration to append to .buckconfig.local
        buckconfig-re = ''

          # ────────────────────────────────────────────────────────────────────────
          # NativeLink Remote Execution (LRE)
          # ────────────────────────────────────────────────────────────────────────
          # Usage: lre-start --workers=${toString cfg.workers} && buck2 build --prefer-remote //:target

          [build]
          execution_platforms = toolchains//:lre

          [buck2_re_client]
          engine_address = grpc://127.0.0.1:${toString cfg.port}
          cas_address = grpc://127.0.0.1:${toString cfg.port}
          action_cache_address = grpc://127.0.0.1:${toString cfg.port}
          tls = false
          instance_name = ${cfg.instance-name}

          [buck2_re_client.platform_properties]
          OSFamily = linux
          container-image = nix-lre-worker
        '';

        # lre-start script
        lre-start = pkgs.writeShellScriptBin "lre-start" ''
          set -euo pipefail

          WORKERS=${toString cfg.workers}
          PORT=${toString cfg.port}
          CONFIG_DIR="''${XDG_RUNTIME_DIR:-/tmp}/nativelink"
          PID_FILE="$CONFIG_DIR/nativelink.pid"
          LOG_FILE="$CONFIG_DIR/nativelink.log"

          usage() {
            echo "Usage: lre-start [OPTIONS]"
            echo ""
            echo "Start NativeLink local remote execution server."
            echo ""
            echo "Options:"
            echo "  --workers=N    Number of worker processes (default: $WORKERS)"
            echo "  --port=N       Port for CAS/scheduler (default: $PORT)"
            echo "  --status       Show status and exit"
            echo "  --stop         Stop running server"
            echo "  --help         Show this help"
          }

          status() {
            if [ -f "$PID_FILE" ]; then
              PID=$(cat "$PID_FILE")
              if kill -0 "$PID" 2>/dev/null; then
                echo "NativeLink running (PID: $PID)"
                echo "  Port: $PORT"
                echo "  Log: $LOG_FILE"
                return 0
              else
                echo "NativeLink not running (stale PID file)"
                rm -f "$PID_FILE"
                return 1
              fi
            else
              echo "NativeLink not running"
              return 1
            fi
          }

          stop() {
            if [ -f "$PID_FILE" ]; then
              PID=$(cat "$PID_FILE")
              if kill -0 "$PID" 2>/dev/null; then
                echo "Stopping NativeLink (PID: $PID)..."
                kill "$PID"
                rm -f "$PID_FILE"
                echo "Stopped."
              else
                echo "NativeLink not running (removing stale PID file)"
                rm -f "$PID_FILE"
              fi
            else
              echo "NativeLink not running"
            fi
          }

          # Parse arguments
          while [[ $# -gt 0 ]]; do
            case "$1" in
              --workers=*) WORKERS="''${1#*=}"; shift ;;
              --port=*) PORT="''${1#*=}"; shift ;;
              --status) status; exit $? ;;
              --stop) stop; exit 0 ;;
              --help|-h) usage; exit 0 ;;
              *) echo "Unknown option: $1"; usage; exit 1 ;;
            esac
          done

          # Check if nativelink is available
          ${
            if nativelink != null then
              ''
                NATIVELINK="${nativelink}/bin/nativelink"
              ''
            else
              ''
                if command -v nativelink &>/dev/null; then
                  NATIVELINK="nativelink"
                else
                  echo "Error: nativelink not found"
                  echo "Run: nix run github:TraceMachina/nativelink -- --help"
                  exit 1
                fi
              ''
          }

          # Create config directory
          mkdir -p "$CONFIG_DIR"

          # Generate config optimized for Buck2
          # Uses compression, proper platform properties, and bytestream
          CONFIG_FILE="$CONFIG_DIR/config.json"
          cat > "$CONFIG_FILE" << NATIVELINK_CONFIG
          {
            "stores": [
              {
                "name": "CAS_MAIN_STORE",
                "compression": {
                  "compression_algorithm": { "lz4": {} },
                  "backend": {
                    "filesystem": {
                      "content_path": "$CONFIG_DIR/cas/content",
                      "temp_path": "$CONFIG_DIR/cas/tmp",
                      "eviction_policy": { "max_bytes": 10737418240 }
                    }
                  }
                }
              },
              {
                "name": "AC_MAIN_STORE",
                "filesystem": {
                  "content_path": "$CONFIG_DIR/ac/content",
                  "temp_path": "$CONFIG_DIR/ac/tmp",
                  "eviction_policy": { "max_bytes": 536870912 }
                }
              }
            ],
            "schedulers": [
              {
                "name": "MAIN_SCHEDULER",
                "simple": {
                  "supported_platform_properties": {
                    "cpu_count": "minimum",
                    "memory_kb": "minimum",
                    "OSFamily": "priority",
                    "container-image": "priority"
                  }
                }
              }
            ],
            "servers": [
              {
                "listener": {
                  "http": { "socket_address": "0.0.0.0:$PORT" }
                },
                "services": {
                  "cas": [{ "cas_store": "CAS_MAIN_STORE" }],
                  "ac": [{ "ac_store": "AC_MAIN_STORE" }],
                  "execution": [{ "cas_store": "CAS_MAIN_STORE", "scheduler": "MAIN_SCHEDULER" }],
                  "capabilities": [{ "remote_execution": { "scheduler": "MAIN_SCHEDULER" } }],
                  "bytestream": { "cas_stores": { "": "CAS_MAIN_STORE" } },
                  "health": {}
                }
              }
            ],
            "global": { "max_open_files": 65536 }
          }
          NATIVELINK_CONFIG

          # Create storage directories
          mkdir -p "$CONFIG_DIR/cas/content" "$CONFIG_DIR/cas/tmp"
          mkdir -p "$CONFIG_DIR/ac/content" "$CONFIG_DIR/ac/tmp"

          # Check if already running
          if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
              echo "NativeLink already running (PID: $PID)"
              exit 0
            fi
            rm -f "$PID_FILE"
          fi

          # Start nativelink
          echo "Starting NativeLink on port $PORT..."
          $NATIVELINK "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
          echo $! > "$PID_FILE"

          # Wait for startup
          sleep 1
          if status >/dev/null 2>&1; then
            echo "NativeLink started successfully"
            echo "  Port: $PORT"
            echo "  PID: $(cat "$PID_FILE")"
            echo ""
            echo "Usage:"
            echo "  buck2 build --prefer-remote //..."
            echo "  buck2 build --remote-only //..."  
          else
            echo "Failed to start NativeLink. Check $LOG_FILE"
            exit 1
          fi
        '';

        # Shell hook to append RE config to .buckconfig.local
        lreShellHook = lib.optionalString isLinux ''
                    # Append RE configuration to .buckconfig.local
                    # (build.nix shellHook creates .buckconfig.local, we append to it)
                    if [ -f ".buckconfig.local" ]; then
                      # Check if RE config already present
                      if ! grep -q "buck2_re_client" .buckconfig.local 2>/dev/null; then
                        cat >> .buckconfig.local << 'LRE_BUCKCONFIG_EOF'
          ${buckconfig-re}
          LRE_BUCKCONFIG_EOF
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
