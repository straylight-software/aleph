-- nix/modules/flake/scripts/lre-buckconfig-remote.dhall
--
-- Remote Execution configuration for Buck2 via Fly.io
-- Supports separate scheduler/CAS endpoints with TLS

let scheduler : Text = env:SCHEDULER as Text
let schedulerPort : Text = env:SCHEDULER_PORT as Text
let cas : Text = env:CAS as Text
let casPort : Text = env:CAS_PORT as Text
let tls : Text = env:TLS as Text
let instanceName : Text = env:INSTANCE_NAME as Text

-- Protocol is passed directly from Nix (grpcs or grpc)
let protocol : Text = env:PROTOCOL as Text

in ''

# ────────────────────────────────────────────────────────────────────────
# Fly.io Remote Execution
# ────────────────────────────────────────────────────────────────────────
# Usage: buck2 build --prefer-remote //:target

[build]
execution_platforms = toolchains//:lre

[buck2_re_client]
engine_address = ${protocol}://${scheduler}:${schedulerPort}
cas_address = ${protocol}://${cas}:${casPort}
action_cache_address = ${protocol}://${cas}:${casPort}
tls = ${tls}
instance_name = ${instanceName}

[buck2_re_client.platform_properties]
OSFamily = linux
container-image = nix-worker
''
