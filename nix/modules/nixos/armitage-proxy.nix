# nix/modules/nixos/armitage-proxy.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                          // armitage-proxy //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The matrix has its roots in primitive arcade games...
#
# Armitage Witness Proxy - TLS MITM proxy for build-time fetch attestation.
#
# Not to block - to **witness**. Every fetch becomes legible:
#   - What was fetched (full URL including path)
#   - When (timestamp)
#   - What came back (SHA256 content hash)
#
# The attestation chain is stored in R2 (`straylight-cas` bucket) for the
# graded monad proof system. Each fetch is a witnessed operation that can
# be replayed and verified.
#
# USAGE:
#
#   services.armitage-proxy = {
#     enable = true;
#     r2 = {
#       enable = true;
#       bucket = "straylight-cas";
#       endpoint = "https://xxx.r2.cloudflarestorage.com";
#       credentials-file = config.age.secrets.r2-env.path;
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
  mk-if = lib.mkIf;
  mk-option = lib.mkOption;
  null-or = lib.types.nullOr;
  to-string = builtins.toString;

  cfg = config.services.armitage-proxy;

  # Proxy URL used throughout

in
{
  _class = "nixos";

  options.services.armitage-proxy = {
    enable = mk-enable-option "Armitage Witness Proxy for build fetch attestation";

    package = mk-option {
      type = lib.types.package;
      default = pkgs.armitage-proxy or (throw "armitage-proxy package not available - add aleph overlay");
      description = "Armitage proxy package to use";
    };

    port = mk-option {
      type = lib.types.port;
      default = 8888;
      description = "Port for the proxy to listen on";
    };

    listen-address = mk-option {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address for the proxy to listen on";
    };

    cache-dir = mk-option {
      type = lib.types.path;
      default = "/var/cache/armitage-proxy";
      description = "Directory for cached fetches (content-addressed by SHA256)";
    };

    log-dir = mk-option {
      type = lib.types.path;
      default = "/var/log/armitage-proxy";
      description = "Directory for attestation logs (JSONL format)";
    };

    cert-dir = mk-option {
      type = lib.types.path;
      default = "/var/lib/armitage-proxy/certs";
      description = "Directory for CA certificate (generated on first run)";
    };

    allowlist = mk-option {
      type = list-of lib.types.str;
      default = [ ];
      description = ''
        Domain allowlist. Empty means allow all.
        Subdomains are automatically included.
      '';
      example = [
        "github.com"
        "githubusercontent.com"
        "crates.io"
        "pypi.org"
      ];
    };

    user = mk-option {
      type = lib.types.str;
      default = "armitage";
      description = "User to run the proxy as";
    };

    group = mk-option {
      type = lib.types.str;
      default = "armitage";
      description = "Group to run the proxy as";
    };

    # R2 attestation sync
    r2 = {
      enable = mk-option {
        type = lib.types.bool;
        default = false;
        description = "Sync attestation logs and cache to Cloudflare R2";
      };

      bucket = mk-option {
        type = lib.types.str;
        default = "straylight-cas";
        description = "R2 bucket name for attestation storage";
      };

      endpoint = mk-option {
        type = lib.types.str;
        default = "";
        description = "R2 endpoint URL";
      };

      key-prefix = mk-option {
        type = lib.types.str;
        default = "attestations/";
        description = "Key prefix for attestation objects in bucket";
      };

      credentials-file = mk-option {
        type = null-or lib.types.path;
        default = null;
        description = "Path to file containing AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY";
      };
    };
  };

  config = mk-if cfg.enable {
    # Create system user/group
    users.users.${cfg.user} = mk-if (cfg.user == "armitage") {
      "isSystemUser" = true;
      inherit (cfg) group;
      description = "Armitage Witness Proxy service user";
      home = cfg.cache-dir;
    };

    users.groups.${cfg.group} = mk-if (cfg.group == "armitage") { };

    # Create directories
    systemd.tmpfiles.rules = [
      "d ${cfg.cache-dir} 0750 ${cfg.user} ${cfg.group} -"
      "d ${cfg.log-dir} 0750 ${cfg.user} ${cfg.group} -"
      "d ${cfg.cert-dir} 0755 ${cfg.user} ${cfg.group} -"
    ];

    # The proxy service
    systemd.services.armitage-proxy = {
      description = "Armitage Witness Proxy";
      "wantedBy" = [ "multi-user.target" ];
      after = [ "network.target" ];

      environment = {
        "PROXY_PORT" = to-string cfg.port;
        "PROXY_CACHE_DIR" = cfg.cache-dir;
        "PROXY_LOG_DIR" = cfg.log-dir;
        "PROXY_CERT_DIR" = cfg.cert-dir;
        "PROXY_ALLOWLIST" = concat-strings-sep "," cfg.allowlist;
      };

      "serviceConfig" = {
        "Type" = "simple";
        "User" = cfg.user;
        "Group" = cfg.group;
        "ExecStart" = get-exe cfg.package;
        "Restart" = "on-failure";
        "RestartSec" = "5s";

        # Hardening
        "NoNewPrivileges" = true;
        "ProtectSystem" = "strict";
        "ProtectHome" = true;
        "PrivateTmp" = true;
        "PrivateDevices" = true;
        "ProtectKernelTunables" = true;
        "ProtectKernelModules" = true;
        "ProtectControlGroups" = true;

        "ReadWritePaths" = [
          cfg.cache-dir
          cfg.log-dir
          cfg.cert-dir
        ];
      };
    };

    # R2 sync service (attestations)
    systemd.services.armitage-proxy-sync = mk-if cfg.r2.enable {
      description = "Sync Armitage attestations to R2";
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      "serviceConfig" = {
        "Type" = "oneshot";
        "User" = cfg.user;
        "Group" = cfg.group;
        "EnvironmentFile" = mk-if (cfg.r2.credentials-file != null) cfg.r2.credentials-file;
        "ExecStart" = get-exe (
          pkgs.writeShellApplication {
            name = "armitage-sync";
            "runtimeInputs" = [ pkgs.awscli2 ];
            text = ''
              # Sync attestation logs
              aws s3 sync \
                --endpoint-url ${cfg.r2.endpoint} \
                ${cfg.log-dir}/ s3://${cfg.r2.bucket}/${cfg.r2.key-prefix}logs/

              # Sync content cache
              aws s3 sync \
                --endpoint-url ${cfg.r2.endpoint} \
                ${cfg.cache-dir}/ s3://${cfg.r2.bucket}/${cfg.r2.key-prefix}cache/
            '';
          }
        );
      };
    };

    # Periodic sync timer
    systemd.timers.armitage-proxy-sync = mk-if cfg.r2.enable {
      description = "Periodic sync of Armitage attestations to R2";
      "wantedBy" = [ "timers.target" ];
      "timerConfig" = {
        "OnCalendar" = "*:0/15"; # Every 15 minutes
        "Persistent" = true;
      };
    };

    # Export CA cert path for other services to use
    environment.variables = {
      "ARMITAGE_CA_CERT" = "${cfg.cert-dir}/ca.pem";
    };
  };
}
