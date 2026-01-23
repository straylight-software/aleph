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
  cfg = config.services.nix-proxy;

  # mitmproxy addon script for caching/logging
  proxyAddon = ./scripts/nix-proxy-addon.py;

  # Proxy URL used throughout
  proxyUrl = "http://${cfg.listenAddress}:${toString cfg.port}";

  # Wrapper script to start mitmproxy with our addon
  proxyScript = pkgs.writeShellApplication {
    name = "nix-proxy";
    runtimeInputs = [ pkgs.mitmproxy ];
    runtimeEnv = {
      NIX_PROXY_CACHE_DIR = "${cfg.cacheDir}";
      NIX_PROXY_LOG_DIR = "${cfg.logDir}";
      NIX_PROXY_ALLOWLIST = lib.concatStringsSep "," cfg.allowlist;
    };
    text = ''
      exec mitmdump \
        --listen-host ${cfg.listenAddress} \
        --listen-port ${toString cfg.port} \
        --set confdir=${cfg.certDir} \
        --scripts ${proxyAddon} \
        ${lib.optionalString cfg.quiet "--quiet"} \
        "$@"
    '';
  };

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
    # Enable the experimental feature required for impure-env
    nix.settings = {
      extra-experimental-features = [ "configurable-impure-env" ];

      # Inject proxy environment into builds
      # This is what actually makes builds use the proxy
      impure-env = [
        "http_proxy=${proxyUrl}"
        "https_proxy=${proxyUrl}"
        "HTTP_PROXY=${proxyUrl}"
        "HTTPS_PROXY=${proxyUrl}"
        "SSL_CERT_FILE=${cfg.certDir}/mitmproxy-ca-cert.pem"
        "NIX_SSL_CERT_FILE=${cfg.certDir}/mitmproxy-ca-cert.pem"
        "CURL_CA_BUNDLE=${cfg.certDir}/mitmproxy-ca-cert.pem"
      ];

      # Make cert dir available inside sandbox for builds that need it
      extra-sandbox-paths = [ cfg.certDir ];
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
        ExecStart = lib.getExe proxyScript;
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
        ExecStart = lib.getExe (
          pkgs.writeShellApplication {
            name = "nix-proxy-sync";
            runtimeInputs = [ pkgs.awscli2 ];
            text = ''
              aws s3 sync \
                --endpoint-url ${cfg.cache.r2.endpoint} \
                ${cfg.cacheDir}/ s3://${cfg.cache.r2.bucket}/
            '';
          }
        );
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

    environment.systemPackages = [ pkgs.mitmproxy ];

    # Nix fetcher uses ssl-cert-file for HTTPS verification (set above in nix.settings)
    # For evaluation-time fetches, the daemon's environment is used.
    # This only affects nix-daemon, not other programs on the system.
    systemd.services.nix-daemon.environment = {
      http_proxy = proxyUrl;
      https_proxy = proxyUrl;
      HTTP_PROXY = proxyUrl;
      HTTPS_PROXY = proxyUrl;
      SSL_CERT_FILE = "${cfg.certDir}/mitmproxy-ca-cert.pem";
      NIX_SSL_CERT_FILE = "${cfg.certDir}/mitmproxy-ca-cert.pem";
      # mkForce needed to override the default set by nix-daemon.nix
      CURL_CA_BUNDLE = lib.mkForce "${cfg.certDir}/mitmproxy-ca-cert.pem";
    };
  };
}
