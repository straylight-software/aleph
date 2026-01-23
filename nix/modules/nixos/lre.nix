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
  cfg = config.services.lre;

  # Scheduler configuration (CAS + AC + Scheduler + optional Worker API)
  schedulerConfig = {
    stores = [
      {
        name = "CAS_MAIN_STORE";
        compression = {
          compression_algorithm.lz4 = { };
          backend =
            if cfg.persistence.enable then
              {
                filesystem = {
                  content_path = "${cfg.persistence.directory}/cas/content";
                  temp_path = "${cfg.persistence.directory}/cas/tmp";
                  eviction_policy.max_bytes = cfg.casMaxBytes;
                };
              }
            else
              {
                memory.eviction_policy.max_bytes = cfg.casMaxBytes;
              };
        };
      }
      {
        name = "AC_MAIN_STORE";
        ${if cfg.persistence.enable then "filesystem" else "memory"} =
          if cfg.persistence.enable then
            {
              content_path = "${cfg.persistence.directory}/ac/content";
              temp_path = "${cfg.persistence.directory}/ac/tmp";
              eviction_policy.max_bytes = cfg.acMaxBytes;
            }
          else
            {
              eviction_policy.max_bytes = cfg.acMaxBytes;
            };
      }
    ];

    schedulers = [
      {
        name = "MAIN_SCHEDULER";
        simple = {
          supported_platform_properties = {
            cpu_count = "minimum";
            memory_kb = "minimum";
            OSFamily = "priority";
            "container-image" = "priority";
          }
          // cfg.platformProperties;
        };
      }
    ];

    servers = [
      # Main API server (CAS, AC, Execution, Capabilities)
      {
        listener.http.socket_address = "${cfg.listenAddress}:${toString cfg.port}";
        services = {
          cas = [ { cas_store = "CAS_MAIN_STORE"; } ];
          ac = [ { ac_store = "AC_MAIN_STORE"; } ];
          execution = [
            {
              cas_store = "CAS_MAIN_STORE";
              scheduler = "MAIN_SCHEDULER";
            }
          ];
          capabilities = [
            {
              remote_execution.scheduler = "MAIN_SCHEDULER";
            }
          ];
          bytestream.cas_stores."" = "CAS_MAIN_STORE";
          health = { };
        };
      }
    ]
    ++ lib.optional cfg.worker.enable {
      # Worker API server (separate port for worker registration)
      listener.http.socket_address = "${cfg.listenAddress}:${toString cfg.worker.apiPort}";
      services = {
        worker_api.scheduler = "MAIN_SCHEDULER";
        health = { };
      };
    };

    global.max_open_files = cfg.maxOpenFiles;
  };

  # Worker configuration
  workerConfig = {
    stores = [
      {
        name = "GRPC_LOCAL_STORE";
        grpc = {
          instance_name = "";
          endpoints = [ { address = "grpc://127.0.0.1:${toString cfg.port}"; } ];
          store_type = "cas";
        };
      }
      {
        name = "GRPC_LOCAL_AC_STORE";
        grpc = {
          instance_name = "";
          endpoints = [ { address = "grpc://127.0.0.1:${toString cfg.port}"; } ];
          store_type = "ac";
        };
      }
      {
        name = "WORKER_FAST_SLOW_STORE";
        fast_slow = {
          fast = {
            filesystem = {
              content_path = "${cfg.persistence.directory}/worker/cas";
              temp_path = "${cfg.persistence.directory}/worker/tmp";
              eviction_policy.max_bytes = cfg.worker.cacheMaxBytes;
            };
          };
          slow.ref_store.name = "GRPC_LOCAL_STORE";
        };
      }
    ];

    workers = [
      {
        local = {
          worker_api_endpoint.uri = "grpc://127.0.0.1:${toString cfg.worker.apiPort}";
          cas_fast_slow_store = "WORKER_FAST_SLOW_STORE";
          upload_action_result.ac_store = "GRPC_LOCAL_AC_STORE";
          work_directory = "${cfg.persistence.directory}/worker/work";
          platform_properties = {
            cpu_count.query_cmd = "nproc";
            memory_kb.query_cmd = "grep MemTotal /proc/meminfo | awk '{print $2}'";
            OSFamily.values = [
              ""
              "linux"
            ];
            "container-image".values = [ "" ];
          }
          // lib.mapAttrs (_: v: { values = [ v ]; }) cfg.worker.platformProperties;
        };
      }
    ];

    servers = [ ];
  };

  schedulerConfigFile = pkgs.writeText "nativelink-scheduler.json" (builtins.toJSON schedulerConfig);

  workerConfigFile = pkgs.writeText "nativelink-worker.json" (builtins.toJSON workerConfig);

in
{
  _class = "nixos";

  options.services.lre = {
    enable = lib.mkEnableOption "NativeLink Local Remote Execution service";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.nativelink or (throw "nativelink package not available");
      defaultText = lib.literalExpression "pkgs.nativelink";
      description = "NativeLink package to use";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 50051;
      description = "Port for NativeLink gRPC services (CAS, AC, execution)";
    };

    listenAddress = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address to listen on. Use 0.0.0.0 for all interfaces.";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open firewall port for NativeLink";
    };

    casMaxBytes = lib.mkOption {
      type = lib.types.int;
      default = 10 * 1024 * 1024 * 1024; # 10 GB
      description = "Maximum size of CAS (Content Addressable Storage) in bytes";
    };

    acMaxBytes = lib.mkOption {
      type = lib.types.int;
      default = 512 * 1024 * 1024; # 512 MB
      description = "Maximum size of AC (Action Cache) in bytes";
    };

    maxOpenFiles = lib.mkOption {
      type = lib.types.int;
      default = 65536;
      description = "Maximum number of open files";
    };

    platformProperties = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Additional platform properties for the scheduler";
    };

    persistence = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Persist CAS and AC to disk (survives restarts)";
      };

      directory = lib.mkOption {
        type = lib.types.path;
        default = "/var/lib/nativelink";
        description = "Directory for persistent storage";
      };
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "nativelink";
      description = "User to run NativeLink as";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "nativelink";
      description = "Group to run NativeLink as";
    };

    # Worker configuration
    worker = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = ''
          Enable local worker for remote execution.
          Without this, NativeLink only provides caching (CAS + AC).
          With this, it can also execute build actions locally.
        '';
      };

      apiPort = lib.mkOption {
        type = lib.types.port;
        default = 50061;
        description = "Port for worker API (worker registration)";
      };

      cacheMaxBytes = lib.mkOption {
        type = lib.types.int;
        default = 5 * 1024 * 1024 * 1024; # 5 GB
        description = "Maximum size of worker's local cache in bytes";
      };

      platformProperties = lib.mkOption {
        type = lib.types.attrsOf lib.types.str;
        default = { };
        description = "Additional platform properties for the worker";
        example = {
          ISA = "x86-64";
          cuda = "12.0";
        };
      };
    };
  };

  config = lib.mkIf cfg.enable {
    users.users.${cfg.user} = lib.mkIf (cfg.user == "nativelink") {
      isSystemUser = true;
      inherit (cfg) group;
      description = "NativeLink service user";
      home = cfg.persistence.directory;
    };

    users.groups.${cfg.group} = lib.mkIf (cfg.group == "nativelink") { };

    systemd.tmpfiles.rules = lib.mkIf cfg.persistence.enable (
      [
        "d ${cfg.persistence.directory} 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/cas 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/cas/content 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/cas/tmp 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/ac 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/ac/content 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/ac/tmp 0750 ${cfg.user} ${cfg.group} -"
      ]
      ++ lib.optionals cfg.worker.enable [
        "d ${cfg.persistence.directory}/worker 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/worker/cas 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/worker/tmp 0750 ${cfg.user} ${cfg.group} -"
        "d ${cfg.persistence.directory}/worker/work 0750 ${cfg.user} ${cfg.group} -"
      ]
    );

    # Main scheduler service
    systemd.services.nativelink = {
      description = "NativeLink Scheduler (CAS/AC/Execution)";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = "${cfg.package}/bin/nativelink ${schedulerConfigFile}";
        Restart = "on-failure";
        RestartSec = "5s";

        # Hardening
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        PrivateDevices = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        LockPersonality = true;

        ReadWritePaths = lib.mkIf cfg.persistence.enable [ cfg.persistence.directory ];
        LimitNOFILE = cfg.maxOpenFiles;
      };
    };

    # Worker service (optional)
    systemd.services.nativelink-worker = lib.mkIf cfg.worker.enable {
      description = "NativeLink Worker";
      wantedBy = [ "multi-user.target" ];
      after = [ "nativelink.service" ];
      requires = [ "nativelink.service" ];

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = "${cfg.package}/bin/nativelink ${workerConfigFile}";
        Restart = "on-failure";
        RestartSec = "10s";

        # Worker needs more permissions to execute builds
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectKernelTunables = true;
        ProtectKernelModules = true;

        ReadWritePaths = [ cfg.persistence.directory ];
        LimitNOFILE = cfg.maxOpenFiles;

        # Give worker time to connect to scheduler
        ExecStartPre = "${pkgs.coreutils}/bin/sleep 2";
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall (
      [ cfg.port ] ++ lib.optional cfg.worker.enable cfg.worker.apiPort
    );

    environment.systemPackages = [ cfg.package ];
  };
}
