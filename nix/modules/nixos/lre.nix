# nix/modules/nixos/lre.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // lre //
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
# NixOS module for Local Remote Execution via NativeLink.
#
# This runs NativeLink as a systemd service for build caching and
# remote execution with Buck2, Bazel, or other RE-compatible clients.
#
# SIMPLE USAGE (cache only):
#
#   services.lre.enable = true;
#
# FULL USAGE (cache + remote execution):
#
#   services.lre = {
#     enable = true;
#     worker.enable = true;  # Enable local worker for execution
#   };
#
# BUCK2 CONFIGURATION (.buckconfig.local):
#
#   [buck2_re_client]
#   engine_address = grpc://127.0.0.1:50051
#   cas_address = grpc://127.0.0.1:50051
#   action_cache_address = grpc://127.0.0.1:50051
#   tls = false
#   instance_name = main
#
# Then build with: buck2 build --prefer-remote //...
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  config,
  lib,
  pkgs,
  ...
}:
let
  # Lisp-case aliases for lib functions
  mk-option = lib.mkOption;
  mk-enable-option = lib.mkEnableOption;
  mk-if = lib.mkIf;
  inherit (lib) optional;
  inherit (lib) optionals;
  map-attrs = lib.mapAttrs;
  literal-expression = lib.literalExpression;
  to-json = builtins.toJSON;
  to-string = builtins.toString;

  cfg = config.services.lre;

  # Scheduler configuration (CAS + AC + Scheduler + optional Worker API)
  # NOTE: Attribute names are quoted because they're NativeLink's JSON schema
  #
  # When R2 is enabled, uses fast_slow store:
  #   - fast: local filesystem (LRU eviction cache)
  #   - slow: Cloudflare R2 (persistent global storage)
  scheduler-config = {
    stores =
      if cfg.r2.enable then
        [
          # Local filesystem as fast tier
          {
            name = "CAS_LOCAL_STORE";
            compression = {
              "compression_algorithm".lz4 = { };
              backend.filesystem = {
                "content_path" = "${cfg.persistence.directory}/cas/content";
                "temp_path" = "${cfg.persistence.directory}/cas/tmp";
                "eviction_policy"."max_bytes" = cfg.cas-max-bytes;
              };
            };
          }
          # R2 as slow tier
          {
            name = "CAS_R2_STORE";
            "experimental_s3_store" = {
              region = "auto";
              inherit (cfg.r2) bucket;
              "key_prefix" = cfg.r2.key-prefix;
              retry = {
                "max_retries" = 6;
                delay = 0.3;
                jitter = 0.5;
              };
            }
            // lib.optionalAttrs (cfg.r2.endpoint != "") {
              "endpoint_url" = cfg.r2.endpoint;
            };
          }
          # Fast/slow composite for CAS
          {
            name = "CAS_MAIN_STORE";
            "fast_slow" = {
              fast."ref_store".name = "CAS_LOCAL_STORE";
              slow."ref_store".name = "CAS_R2_STORE";
            };
          }
          # AC store (local only - action cache doesn't need global persistence)
          {
            name = "AC_MAIN_STORE";
            filesystem = {
              "content_path" = "${cfg.persistence.directory}/ac/content";
              "temp_path" = "${cfg.persistence.directory}/ac/tmp";
              "eviction_policy"."max_bytes" = cfg.ac-max-bytes;
            };
          }
        ]
      else
        [
          {
            name = "CAS_MAIN_STORE";
            compression = {
              "compression_algorithm".lz4 = { };
              backend =
                if cfg.persistence.enable then
                  {
                    filesystem = {
                      "content_path" = "${cfg.persistence.directory}/cas/content";
                      "temp_path" = "${cfg.persistence.directory}/cas/tmp";
                      "eviction_policy"."max_bytes" = cfg.cas-max-bytes;
                    };
                  }
                else
                  {
                    memory."eviction_policy"."max_bytes" = cfg.cas-max-bytes;
                  };
            };
          }
          {
            name = "AC_MAIN_STORE";
            ${if cfg.persistence.enable then "filesystem" else "memory"} =
              if cfg.persistence.enable then
                {
                  "content_path" = "${cfg.persistence.directory}/ac/content";
                  "temp_path" = "${cfg.persistence.directory}/ac/tmp";
                  "eviction_policy"."max_bytes" = cfg.ac-max-bytes;
                }
              else
                {
                  "eviction_policy"."max_bytes" = cfg.ac-max-bytes;
                };
          }
        ];

    schedulers = [
      {
        name = "MAIN_SCHEDULER";
        simple = {
          "supported_platform_properties" = {
            "cpu_count" = "minimum";
            "memory_kb" = "minimum";
            "OSFamily" = "priority";
            "container-image" = "priority";
          }
          // cfg.platform-properties;
        };
      }
    ];

    servers = [
      # Main API server (CAS, AC, Execution, Capabilities)
      {
        listener.http."socket_address" = "${cfg.listen-address}:${to-string cfg.port}";
        services = {
          cas = [ { "cas_store" = "CAS_MAIN_STORE"; } ];
          ac = [ { "ac_store" = "AC_MAIN_STORE"; } ];
          execution = [
            {
              "cas_store" = "CAS_MAIN_STORE";
              scheduler = "MAIN_SCHEDULER";
            }
          ];
          capabilities = [
            {
              "remote_execution".scheduler = "MAIN_SCHEDULER";
            }
          ];
          bytestream."cas_stores"."" = "CAS_MAIN_STORE";
          health = { };
        };
      }
    ]
    ++ optional cfg.worker.enable {
      # Worker API server (separate port for worker registration)
      listener.http."socket_address" = "${cfg.listen-address}:${to-string cfg.worker.api-port}";
      services = {
        "worker_api".scheduler = "MAIN_SCHEDULER";
        health = { };
      };
    };

    global."max_open_files" = cfg.max-open-files;
  };

  # Worker configuration
  # NOTE: Attribute names are quoted because they're NativeLink's JSON schema
  worker-config = {
    stores = [
      {
        name = "GRPC_LOCAL_STORE";
        grpc = {
          "instance_name" = "";
          endpoints = [ { address = "grpc://127.0.0.1:${to-string cfg.port}"; } ];
          "store_type" = "cas";
        };
      }
      {
        name = "GRPC_LOCAL_AC_STORE";
        grpc = {
          "instance_name" = "";
          endpoints = [ { address = "grpc://127.0.0.1:${to-string cfg.port}"; } ];
          "store_type" = "ac";
        };
      }
      {
        name = "WORKER_FAST_SLOW_STORE";
        "fast_slow" = {
          fast = {
            filesystem = {
              "content_path" = "${cfg.persistence.directory}/worker/cas";
              "temp_path" = "${cfg.persistence.directory}/worker/tmp";
              "eviction_policy"."max_bytes" = cfg.worker.cache-max-bytes;
            };
          };
          slow."ref_store".name = "GRPC_LOCAL_STORE";
        };
      }
    ];

    workers = [
      {
        local = {
          "worker_api_endpoint".uri = "grpc://127.0.0.1:${to-string cfg.worker.api-port}";
          "cas_fast_slow_store" = "WORKER_FAST_SLOW_STORE";
          "upload_action_result"."ac_store" = "GRPC_LOCAL_AC_STORE";
          "work_directory" = "${cfg.persistence.directory}/worker/work";
          "platform_properties" = {
            # nproc works without shell - simple single command
            "cpu_count"."query_cmd" = "nproc";
            # NativeLink runs query_cmd without a shell, so pipes/awk don't work.
            # Use static value based on common server memory. Workers with different
            # memory can override via worker.platform-properties.
            "memory_kb".values = [ "32000000" ]; # ~32GB default
            "OSFamily".values = [
              ""
              "linux"
            ];
            # Accept both empty (legacy) and nix-worker (matches Fly.io workers)
            "container-image".values = [
              ""
              "nix-worker"
            ];
          }
          // map-attrs (_: v: { values = [ v ]; }) cfg.worker.platform-properties;
        };
      }
    ];

    servers = [ ];
  };

  scheduler-config-file = pkgs.writeText "nativelink-scheduler.json" (to-json scheduler-config);

  worker-config-file = pkgs.writeText "nativelink-worker.json" (to-json worker-config);

in
{
  _class = "nixos";

  options.services.lre = {
    enable = mk-enable-option "NativeLink Local Remote Execution service";

    package = mk-option {
      type = lib.types.package;
      default = pkgs.nativelink or (throw "nativelink package not available");
      "defaultText" = literal-expression "pkgs.nativelink";
      description = "NativeLink package to use";
    };

    port = mk-option {
      type = lib.types.port;
      default = 50051;
      description = "Port for NativeLink gRPC services (CAS, AC, execution)";
    };

    listen-address = mk-option {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address to listen on. Use 0.0.0.0 for all interfaces.";
    };

    open-firewall = mk-option {
      type = lib.types.bool;
      default = false;
      description = "Open firewall port for NativeLink";
    };

    cas-max-bytes = mk-option {
      type = lib.types.int;
      default = 10 * 1024 * 1024 * 1024; # 10 GB
      description = "Maximum size of CAS (Content Addressable Storage) in bytes";
    };

    ac-max-bytes = mk-option {
      type = lib.types.int;
      default = 512 * 1024 * 1024; # 512 MB
      description = "Maximum size of AC (Action Cache) in bytes";
    };

    max-open-files = mk-option {
      type = lib.types.int;
      default = 65536;
      description = "Maximum number of open files";
    };

    platform-properties = mk-option {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Additional platform properties for the scheduler";
    };

    persistence = {
      enable = mk-option {
        type = lib.types.bool;
        default = true;
        description = "Persist CAS and AC to disk (survives restarts)";
      };

      directory = mk-option {
        type = lib.types.path;
        default = "/var/lib/nativelink";
        description = "Directory for persistent storage";
      };
    };

    # R2 backend for global persistent storage
    r2 = {
      enable = mk-option {
        type = lib.types.bool;
        default = false;
        description = ''
          Enable Cloudflare R2 as slow tier backend for CAS.
          Local filesystem becomes a fast LRU cache, R2 provides
          persistent global storage shared across machines.

          Requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
          environment variables (R2 uses S3-compatible API).
        '';
      };

      bucket = mk-option {
        type = lib.types.str;
        default = "nativelink-cas";
        description = "R2 bucket name";
      };

      endpoint = mk-option {
        type = lib.types.str;
        default = "";
        description = "R2 S3-compatible endpoint URL (e.g., https://<account>.r2.cloudflarestorage.com)";
      };

      key-prefix = mk-option {
        type = lib.types.str;
        default = "cas/";
        description = "Key prefix for objects in bucket";
      };

      credentials-file = mk-option {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = ''
          Path to file containing R2 credentials as environment variables:
            AWS_ACCESS_KEY_ID=...
            AWS_SECRET_ACCESS_KEY=...

          If null, credentials must be provided via EnvironmentFile in systemd
          or through another mechanism.
        '';
      };
    };

    user = mk-option {
      type = lib.types.str;
      default = "nativelink";
      description = "User to run NativeLink as";
    };

    group = mk-option {
      type = lib.types.str;
      default = "nativelink";
      description = "Group to run NativeLink as";
    };

    # Worker configuration
    worker = {
      enable = mk-option {
        type = lib.types.bool;
        default = false;
        description = ''
          Enable local worker for remote execution.
          Without this, NativeLink only provides caching (CAS + AC).
          With this, it can also execute build actions locally.
        '';
      };

      api-port = mk-option {
        type = lib.types.port;
        default = 50061;
        description = "Port for worker API (worker registration)";
      };

      cache-max-bytes = mk-option {
        type = lib.types.int;
        default = 5 * 1024 * 1024 * 1024; # 5 GB
        description = "Maximum size of worker's local cache in bytes";
      };

      platform-properties = mk-option {
        type = lib.types.attrsOf lib.types.str;
        default = { };
        description = "Additional platform properties for the worker";
        example = {
          "ISA" = "x86-64";
          cuda = "12.0";
        };
      };
    };
  };

  # NOTE: NixOS module attributes are quoted because they're external API
  config = mk-if cfg.enable {
    users.users.${cfg.user} = mk-if (cfg.user == "nativelink") {
      "isSystemUser" = true;
      inherit (cfg) group;
      description = "NativeLink service user";
      home = cfg.persistence.directory;
    };

    users.groups.${cfg.group} = mk-if (cfg.group == "nativelink") { };

    systemd.tmpfiles.rules = mk-if cfg.persistence.enable (
      [
        "d ${cfg.persistence.directory} 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/cas 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/cas/content 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/cas/tmp 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/ac 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/ac/content 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/ac/tmp 0750 ${cfg.user} ${cfg.group} -"
      ]
      ++ optionals cfg.worker.enable [
        "d ${cfg.persistence.directory}/worker 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/worker/cas 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/worker/tmp 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/worker/work 0750 ${cfg.user} ${cfg.group} -"
      ]
    );

    # Main scheduler service
    systemd.services.nativelink = {
      description = "NativeLink Scheduler (CAS/AC/Execution)";
      "wantedBy" = [ "multi-user.target" ];
      after = [ "network.target" ];

      "serviceConfig" = {
        "Type" = "simple";
        "User" = cfg.user;
        "Group" = cfg.group;
        "ExecStart" = "${cfg.package}/bin/nativelink ${scheduler-config-file}";
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
        "RestrictSUIDSGID" = true;
        "LockPersonality" = true;

        "ReadWritePaths" = mk-if cfg.persistence.enable [ cfg.persistence.directory ];
        "LimitNOFILE" = cfg.max-open-files;
      }
      # R2 credentials from file (if configured)
      // lib.optionalAttrs (cfg.r2.enable && cfg.r2.credentials-file != null) {
        "EnvironmentFile" = cfg.r2.credentials-file;
      };
    };

    # Worker service (optional)
    systemd.services.nativelink-worker = mk-if cfg.worker.enable {
      description = "NativeLink Worker";
      "wantedBy" = [ "multi-user.target" ];
      after = [ "nativelink.service" ];
      requires = [ "nativelink.service" ];

      "serviceConfig" = {
        "Type" = "simple";
        "User" = cfg.user;
        "Group" = cfg.group;
        "ExecStart" = "${cfg.package}/bin/nativelink ${worker-config-file}";
        "Restart" = "on-failure";
        "RestartSec" = "10s";

        # Worker needs more permissions to execute builds
        "NoNewPrivileges" = true;
        "ProtectSystem" = "strict";
        "ProtectKernelTunables" = true;
        "ProtectKernelModules" = true;

        "ReadWritePaths" = [ cfg.persistence.directory ];
        "LimitNOFILE" = cfg.max-open-files;

        # Give worker time to connect to scheduler
        "ExecStartPre" = "${pkgs.coreutils}/bin/sleep 2";
      };
    };

    networking.firewall."allowedTCPPorts" = mk-if cfg.open-firewall (
      [ cfg.port ] ++ optional cfg.worker.enable cfg.worker.api-port
    );

    environment."systemPackages" = [ cfg.package ];
  };
}
