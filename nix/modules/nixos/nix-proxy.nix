# nix/modules/nixos/nix-proxy.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                            // nix-proxy //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "The matrix has its roots in primitive arcade games..."
#
# Replaces Nix's broken sandbox network blocking with a proper proxy-based
# approach. All network traffic from nix-daemon goes through mitmproxy,
# which can cache, log, and enforce policy.
#
# The Nix sandbox's network isolation (unshare CLONE_NEWNET) is disabled
# because it's security theater - FODs get network anyway, and everything
# else is just broken. Instead, we route through a proxy that:
#
#   1. Caches fetched content (content-addressed, to R2)
#   2. Logs all fetches (attestation)
#   3. Enforces allowlist policy
#   4. Actually works
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
  cfg = config.services.nix-proxy;

  # mitmproxy addon script for caching/logging
  proxyAddon = ./scripts/nix-proxy-addon.py;

  # Wrapper script to start mitmproxy with our addon
  proxyScript = pkgs.writeShellScript "nix-proxy" ''
    set -euo pipefail

    export NIX_PROXY_CACHE_DIR="${cfg.cacheDir}"
    export NIX_PROXY_LOG_DIR="${cfg.logDir}"
    export NIX_PROXY_ALLOWLIST="${lib.concatStringsSep "," cfg.allowlist}"

    exec ${pkgs.mitmproxy}/bin/mitmdump \
      --listen-host ${cfg.listenAddress} \
      --listen-port ${toString cfg.port} \
      --set confdir=${cfg.certDir} \
      --scripts ${proxyAddon} \
      ${lib.optionalString cfg.quiet "--quiet"} \
      "$@"
  '';

in
{
  _class = "nixos";

  options.services.nix-proxy = {
    enable = lib.mkEnableOption "Nix network proxy for controlled fetching";

    port = lib.mkOption {
      type = lib.types.port;
      default = 8080;
      description = "Port for the proxy to listen on";
    };

    listenAddress = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address for the proxy to listen on";
    };

    cacheDir = lib.mkOption {
      type = lib.types.path;
      default = "/var/cache/nix-proxy";
      description = "Directory for cached fetches (content-addressed)";
    };

    logDir = lib.mkOption {
      type = lib.types.path;
      default = "/var/log/nix-proxy";
      description = "Directory for fetch logs";
    };

    certDir = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/nix-proxy/certs";
      description = "Directory for mitmproxy CA certificate";
    };

    allowlist = lib.mkOption {
      type = lib.types.listOf lib.types.str;
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

    quiet = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Suppress mitmproxy output (logs still written to logDir)";
    };

    # R2 cache sync (optional)
    cache.r2 = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Sync cache to Cloudflare R2";
      };

      bucket = lib.mkOption {
        type = lib.types.str;
        default = "";
        description = "R2 bucket name";
      };

      endpoint = lib.mkOption {
        type = lib.types.str;
        default = "";
        description = "R2 endpoint URL";
      };

      credentialsFile = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = "Path to file containing AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    # Relax Nix sandbox - we're replacing its network blocking with the proxy
    nix.settings = {
      # Allow network in sandbox (we control it via proxy)
      sandbox = "relaxed";

      # Trust our CA for HTTPS interception
      ssl-cert-file = "${cfg.certDir}/mitmproxy-ca-cert.pem";

      # Ensure cert is available in sandbox
      extra-sandbox-paths = [ cfg.certDir ];
    };

    # Set proxy via environment for nix-daemon
    systemd.services.nix-daemon.environment = {
      http_proxy = "http://${cfg.listenAddress}:${toString cfg.port}";
      https_proxy = "http://${cfg.listenAddress}:${toString cfg.port}";
      HTTP_PROXY = "http://${cfg.listenAddress}:${toString cfg.port}";
      HTTPS_PROXY = "http://${cfg.listenAddress}:${toString cfg.port}";
      SSL_CERT_FILE = "${cfg.certDir}/mitmproxy-ca-cert.pem";
      NIX_SSL_CERT_FILE = "${cfg.certDir}/mitmproxy-ca-cert.pem";
      # Override CURL_CA_BUNDLE set by nix-daemon.nix / Determinate Nix
      CURL_CA_BUNDLE = lib.mkForce "${cfg.certDir}/mitmproxy-ca-cert.pem";
    };

    # Create directories
    # certDir needs 0755 so nix-daemon can read the CA cert
    systemd.tmpfiles.rules = [
      "d ${cfg.cacheDir} 0755 root root -"
      "d ${cfg.logDir} 0755 root root -"
      "d ${cfg.certDir} 0755 root root -"
    ];

    # Generate CA cert on first boot
    system.activationScripts.nix-proxy-cert = lib.stringAfter [ "var" ] ''
      if [ ! -f "${cfg.certDir}/mitmproxy-ca-cert.pem" ]; then
        echo "Generating mitmproxy CA certificate..."
        ${pkgs.mitmproxy}/bin/mitmdump --set confdir=${cfg.certDir} -q &
        PID=$!
        sleep 2
        kill $PID 2>/dev/null || true
        # Wait for cert to be written
        for i in $(seq 1 10); do
          [ -f "${cfg.certDir}/mitmproxy-ca-cert.pem" ] && break
          sleep 0.5
        done
      fi
    '';

    # The proxy service
    systemd.services.nix-proxy = {
      description = "Nix Network Proxy";
      wantedBy = [ "multi-user.target" ];
      before = [ "nix-daemon.service" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];

      serviceConfig = {
        Type = "simple";
        ExecStart = proxyScript;
        Restart = "on-failure";
        RestartSec = "5s";

        # Hardening
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        ReadWritePaths = [
          cfg.cacheDir
          cfg.logDir
          cfg.certDir
        ];
      };
    };

    # Optional: R2 sync timer
    systemd.services.nix-proxy-sync = lib.mkIf cfg.cache.r2.enable {
      description = "Sync Nix proxy cache to R2";
      serviceConfig = {
        Type = "oneshot";
        EnvironmentFile = lib.mkIf (cfg.cache.r2.credentialsFile != null) cfg.cache.r2.credentialsFile;
        ExecStart = pkgs.writeShellScript "nix-proxy-sync" ''
          ${pkgs.awscli2}/bin/aws s3 sync \
            --endpoint-url ${cfg.cache.r2.endpoint} \
            ${cfg.cacheDir}/ s3://${cfg.cache.r2.bucket}/
        '';
      };
    };

    systemd.timers.nix-proxy-sync = lib.mkIf cfg.cache.r2.enable {
      description = "Periodic sync of Nix proxy cache to R2";
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = "hourly";
        Persistent = true;
      };
    };

    # Add package to system for CLI access
    environment.systemPackages = [ pkgs.mitmproxy ];

    # Set proxy env vars system-wide so nix client uses proxy too
    # (not just daemon - flake fetching happens in client)
    environment.sessionVariables = {
      http_proxy = "http://${cfg.listenAddress}:${toString cfg.port}";
      https_proxy = "http://${cfg.listenAddress}:${toString cfg.port}";
      HTTP_PROXY = "http://${cfg.listenAddress}:${toString cfg.port}";
      HTTPS_PROXY = "http://${cfg.listenAddress}:${toString cfg.port}";
      SSL_CERT_FILE = "${cfg.certDir}/mitmproxy-ca-cert.pem";
      NIX_SSL_CERT_FILE = "${cfg.certDir}/mitmproxy-ca-cert.pem";
      CURL_CA_BUNDLE = "${cfg.certDir}/mitmproxy-ca-cert.pem";
    };
  };
}
