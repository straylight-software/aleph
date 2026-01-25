# nix/modules/nixos/nix-proxy.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                            // nix-proxy //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "The matrix has its roots in primitive arcade games..."
#
# Routes Nix build network traffic through mitmproxy for caching, logging,
# and policy enforcement. This is for builds that need CAS server access
# (non-sandboxed builds, FODs, etc).
#
# Uses Nix's `impure-env` to inject proxy environment variables directly
# into build environments. This requires the `configurable-impure-env`
# experimental feature.
#
# The proxy:
#   1. Caches fetched content (content-addressed, syncs to R2)
#   2. Logs all fetches (attestation)
#   3. Enforces domain allowlist policy
#
# USAGE:
#
#   services.nix-proxy = {
#     enable = true;
#     cache.r2 = {
#       bucket = "nix-cache";
#       endpoint = "https://xxx.r2.cloudflarestorage.com";
#     };
#   };
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  config,
  lib,
  pkgs,
  ...
}:
let
  # lisp-case aliases for lib functions
  concat-strings-sep = lib.concatStringsSep;
  get-exe = lib.getExe;
  list-of = lib.types.listOf;
  mk-enable-option = lib.mkEnableOption;
  mk-force = lib.mkForce;
  mk-if = lib.mkIf;
  mk-option = lib.mkOption;
  null-or = lib.types.nullOr;
  optional-string = lib.optionalString;
  string-after = lib.stringAfter;
  to-string = builtins.toString;

  cfg = config.services.nix-proxy;

  # mitmproxy addon script for caching/logging
  proxy-addon = ./scripts/nix-proxy-addon.py;

  # Proxy URL used throughout
  proxy-url = "http://${cfg.listen-address}:${to-string cfg.port}";

  # Wrapper script to start mitmproxy with our addon
  # NOTE: writeShellApplication attrs are quoted - external API
  proxy-script = pkgs.writeShellApplication {
    name = "nix-proxy";
    "runtimeInputs" = [ pkgs.mitmproxy ];
    "runtimeEnv" = {
      "NIX_PROXY_CACHE_DIR" = "${cfg.cache-dir}";
      "NIX_PROXY_LOG_DIR" = "${cfg.log-dir}";
      "NIX_PROXY_ALLOWLIST" = concat-strings-sep "," cfg.allowlist;
    };
    text = ''
      exec mitmdump \
        --listen-host ${cfg.listen-address} \
        --listen-port ${to-string cfg.port} \
        --set confdir=${cfg.cert-dir} \
        --scripts ${proxy-addon} \
        ${optional-string cfg.quiet "--quiet"} \
        "$@"
    '';
  };

in
{
  _class = "nixos";

  options.services.nix-proxy = {
    enable = mk-enable-option "Nix network proxy for controlled fetching";

    port = mk-option {
      type = lib.types.port;
      default = 8080;
      description = "Port for the proxy to listen on";
    };

    listen-address = mk-option {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address for the proxy to listen on";
    };

    cache-dir = mk-option {
      type = lib.types.path;
      default = "/var/cache/nix-proxy";
      description = "Directory for cached fetches (content-addressed)";
    };

    log-dir = mk-option {
      type = lib.types.path;
      default = "/var/log/nix-proxy";
      description = "Directory for fetch logs";
    };

    cert-dir = mk-option {
      type = lib.types.path;
      default = "/var/lib/nix-proxy/certs";
      description = "Directory for mitmproxy CA certificate";
    };

    allowlist = mk-option {
      type = list-of lib.types.str;
      default = [ ];
      description = ''
        Domain allowlist. Empty means allow all.
        Subdomains are automatically included (e.g., "github.com" allows "raw.githubusercontent.com").
      '';
      example = [
        "github.com"
        "githubusercontent.com"
        "nixos.org"
        "cache.nixos.org"
        "releases.nixos.org"
        "tarballs.nixos.org"
        "crates.io"
        "static.crates.io"
        "pypi.org"
        "files.pythonhosted.org"
      ];
    };

    quiet = mk-option {
      type = lib.types.bool;
      default = true;
      description = "Suppress mitmproxy output (logs still written to log-dir)";
    };

    # R2 cache sync (optional)
    cache.r2 = {
      enable = mk-option {
        type = lib.types.bool;
        default = false;
        description = "Sync cache to Cloudflare R2";
      };

      bucket = mk-option {
        type = lib.types.str;
        default = "";
        description = "R2 bucket name";
      };

      endpoint = mk-option {
        type = lib.types.str;
        default = "";
        description = "R2 endpoint URL";
      };

      credentials-file = mk-option {
        type = null-or lib.types.path;
        default = null;
        description = "Path to file containing AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY";
      };
    };
  };

  # NOTE: NixOS module config attributes are quoted - external API
  config = mk-if cfg.enable {
    # Enable the experimental feature required for impure-env
    nix.settings = {
      extra-experimental-features = [ "configurable-impure-env" ];

      # Inject proxy environment into builds
      # This is what actually makes builds use the proxy
      impure-env = [
        "http_proxy=${proxy-url}"
        "https_proxy=${proxy-url}"
        "HTTP_PROXY=${proxy-url}"
        "HTTPS_PROXY=${proxy-url}"
        "SSL_CERT_FILE=${cfg.cert-dir}/mitmproxy-ca-cert.pem"
        "NIX_SSL_CERT_FILE=${cfg.cert-dir}/mitmproxy-ca-cert.pem"
        "CURL_CA_BUNDLE=${cfg.cert-dir}/mitmproxy-ca-cert.pem"
      ];

      # Make cert dir available inside sandbox for builds that need it
      extra-sandbox-paths = [ cfg.cert-dir ];
    };

    # Create directories
    # cert-dir needs 0755 so nix-daemon can read the CA cert
    systemd.tmpfiles.rules = [
      "d ${cfg.cache-dir} 0755 root root -"
      "d ${cfg.log-dir} 0755 root root -"
      "d ${cfg.cert-dir} 0755 root root -"
    ];

    # Generate CA cert on first boot
    system."activationScripts".nix-proxy-cert = string-after [ "var" ] ''
      if [ ! -f "${cfg.cert-dir}/mitmproxy-ca-cert.pem" ]; then
        echo "Generating mitmproxy CA certificate..."
        ${pkgs.mitmproxy}/bin/mitmdump --set confdir=${cfg.cert-dir} -q &
        PID=$!
        sleep 2
        kill $PID 2>/dev/null || true
        # Wait for cert to be written
        for i in $(seq 1 10); do
          [ -f "${cfg.cert-dir}/mitmproxy-ca-cert.pem" ] && break
          sleep 0.5
        done
      fi
    '';

    # The proxy service
    systemd.services.nix-proxy = {
      description = "Nix Network Proxy";
      "wantedBy" = [ "multi-user.target" ];
      before = [ "nix-daemon.service" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];

      "serviceConfig" = {
        "Type" = "simple";
        "ExecStart" = get-exe proxy-script;
        "Restart" = "on-failure";
        "RestartSec" = "5s";

        # Hardening
        "NoNewPrivileges" = true;
        "ProtectSystem" = "strict";
        "ProtectHome" = true;
        "PrivateTmp" = true;
        "ReadWritePaths" = [
          cfg.cache-dir
          cfg.log-dir
          cfg.cert-dir
        ];
      };
    };

    # Optional: R2 sync timer
    systemd.services.nix-proxy-sync = mk-if cfg.cache.r2.enable {
      description = "Sync Nix proxy cache to R2";
      "serviceConfig" = {
        "Type" = "oneshot";
        "EnvironmentFile" = mk-if (cfg.cache.r2.credentials-file != null) cfg.cache.r2.credentials-file;
        "ExecStart" = get-exe (
          pkgs.writeShellApplication {
            name = "nix-proxy-sync";
            "runtimeInputs" = [ pkgs.awscli2 ];
            text = ''
              aws s3 sync \
                --endpoint-url ${cfg.cache.r2.endpoint} \
                ${cfg.cache-dir}/ s3://${cfg.cache.r2.bucket}/
            '';
          }
        );
      };
    };

    systemd.timers.nix-proxy-sync = mk-if cfg.cache.r2.enable {
      description = "Periodic sync of Nix proxy cache to R2";
      "wantedBy" = [ "timers.target" ];
      "timerConfig" = {
        "OnCalendar" = "hourly";
        "Persistent" = true;
      };
    };

    environment."systemPackages" = [ pkgs.mitmproxy ];

    # Nix fetcher uses ssl-cert-file for HTTPS verification (set above in nix.settings)
    # For evaluation-time fetches, the daemon's environment is used.
    # This only affects nix-daemon, not other programs on the system.
    systemd.services.nix-daemon.environment = {
      "http_proxy" = proxy-url;
      "https_proxy" = proxy-url;
      "HTTP_PROXY" = proxy-url;
      "HTTPS_PROXY" = proxy-url;
      "SSL_CERT_FILE" = "${cfg.cert-dir}/mitmproxy-ca-cert.pem";
      "NIX_SSL_CERT_FILE" = "${cfg.cert-dir}/mitmproxy-ca-cert.pem";
      # mkForce needed to override the default set by nix-daemon.nix
      "CURL_CA_BUNDLE" = mk-force "${cfg.cert-dir}/mitmproxy-ca-cert.pem";
    };
  };
}
